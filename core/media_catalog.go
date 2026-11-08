// Built-in, schema-versioned catalog of media generation model packages.
// The catalog is human-reviewed data shipped with the app (embedded JSON):
// it pins the exact component identity (source repo, revision, file, byte
// size, sha256) for every downloadable/importable artifact so downloads and
// imports are verifiable, and so a "ready" install claim is anchored to
// known bytes rather than file names (plan §5.2/§5.3).
//
// The catalog is deliberately small and additive: adding a model means
// adding one JSON entry plus, when needed, a runner profile. Nothing here
// knows about llama.cpp.
package core

import (
	"embed"
	"encoding/json"
	"fmt"
	"sync"
)

// mediaCatalogFS embeds the catalog JSON documents. One file per model
// package keeps reviews self-contained; the loader reads them all.
//
//go:embed media_catalog/*.json
var mediaCatalogFS embed.FS

// Catalog schema version. Bump when a field semantic changes in a way old
// pinned data cannot express; loaders reject catalogs with a different
// version rather than guessing.
const mediaCatalogSchemaVersion = 1

// Media component roles. The set is open-ended by design (diffusion model,
// high-noise diffusion, text encoder, vision encoder, video VAE, audio VAE
// — plan §5.2) but each catalog entry must use a role from this list so a
// typo cannot silently wire a component to nowhere. The runner profile
// interprets the roles it knows and rejects variants carrying roles it
// cannot wire.
const (
	MediaRoleDiffusion    = "diffusion"
	MediaRoleTextEncoder  = "text_encoder"
	MediaRoleVAE          = "vae"
	MediaRoleHighNoiseDif = "high_noise_diffusion"
	MediaRoleVisionEnc    = "vision_encoder"
	MediaRoleVideoVAE     = "video_vae"
	MediaRoleAudioVAE     = "audio_vae"
)

// mediaKnownRoles is the validation allow-list of component roles.
var mediaKnownRoles = map[string]bool{
	MediaRoleDiffusion:    true,
	MediaRoleTextEncoder:  true,
	MediaRoleVAE:          true,
	MediaRoleHighNoiseDif: true,
	MediaRoleVisionEnc:    true,
	MediaRoleVideoVAE:     true,
	MediaRoleAudioVAE:     true,
}

// MediaComponent pins the identity of one downloadable weight file: where it
// came from (Repo@Revision), what it is called there, its exact size and
// content hash. Two components with the same Repo+Revision+File are the same
// asset and must not be downloaded twice (dedupe rule, plan §5.3).
type MediaComponent struct {
	Role     string `json:"role"`     // one of the MediaRole* constants
	Repo     string `json:"repo"`     // source repo id, e.g. "leejet/Z-Image-Turbo-GGUF"
	Revision string `json:"revision"` // pinned commit/ref of Repo
	File     string `json:"file"`     // file name (or repo-relative path) within Repo@Revision
	Size     int64  `json:"size"`     // exact byte size; 0 means "unknown yet" (shown as such)
	SHA256   string `json:"sha256"`   // lowercase hex; empty means "unknown yet" (blocks readiness)
	Format   string `json:"format"`   // gguf / safetensors (informational + import hint)
}

// Identity returns the dedupe key of the component asset.
func (c MediaComponent) Identity() string {
	return c.Repo + "@" + c.Revision + ":" + c.File
}

// MediaRunnerProfile describes how a runner turns a variant plus a validated
// job request into an engine command. P1 ships the fixed image profile; the
// field stays extensible for video profiles later.
type MediaRunnerProfile struct {
	// Family names the engine wiring, e.g. "z-image-turbo". The runner
	// dispatches on it; unknown families fail closed.
	Family string `json:"family"`
	// Kind is the capability the profile implements: "image" (P1) or
	// "video" (later; rejected while unverified, plan §7 P1).
	Kind string `json:"kind"`
	// Defaults are profile-level generation defaults (cfg scale, steps,
	// sampling method) the job request may override within validated ranges.
	CFGScale     float64 `json:"cfgScale"`
	Steps        int     `json:"steps"`
	Sampler      string  `json:"sampler"`
	MinSize      int     `json:"minSize"` // inclusive W/H bounds for the request
	MaxSize      int     `json:"maxSize"`
	SizeMultiple int     `json:"sizeMultiple"` // W/H must be a multiple of this (latent alignment)
}

// MediaVariant is one concrete, installable flavor of a model: a quantization
// choice with its exact component set. The Label may name the main model's
// quant but component precision is per-component (plan §2.1 risk 5).
type MediaVariant struct {
	ID         string             `json:"id"`
	Label      string             `json:"label"`
	Components []MediaComponent   `json:"components"`
	Profile    MediaRunnerProfile `json:"profile"`
}

// MediaModel is a catalog entry: a model family with its variants.
type MediaModel struct {
	ID          string         `json:"id"`
	Family      string         `json:"family"`
	Name        string         `json:"name"`
	Developer   string         `json:"developer"`
	License     string         `json:"license"`
	Description string         `json:"description"` // reviewed bilingual-free neutral copy (zh/en rendered by i18n keys elsewhere)
	Variants    []MediaVariant `json:"variants"`
}

// mediaCatalog is the parsed and validated in-memory catalog.
type mediaCatalog struct {
	SchemaVersion int          `json:"schemaVersion"`
	Models        []MediaModel `json:"models"`
}

var (
	mediaCatalogOnce   sync.Once
	mediaCatalogLoaded *mediaCatalog
	mediaCatalogErr    error
	// mediaCatalogRead is the injection seam for tests (reads the embedded
	// catalog file in production).
	mediaCatalogRead = func() ([]byte, error) {
		return mediaCatalogFS.ReadFile("media_catalog/catalog.json")
	}
)

// GetMediaCatalog parses and validates the embedded catalog once per process.
// A broken catalog is a build-time data bug: it surfaces as an error here and
// every media entry point fails closed on it.
func GetMediaCatalog() (*mediaCatalog, error) {
	mediaCatalogOnce.Do(func() {
		mediaCatalogLoaded, mediaCatalogErr = loadMediaCatalog(mediaCatalogRead)
	})
	return mediaCatalogLoaded, mediaCatalogErr
}

// loadMediaCatalog reads, parses and validates catalog data from the injected
// reader. Validation is fail-closed: any structural problem (version mismatch,
// duplicate ids, unknown roles, dangling variant references, bad hashes)
// rejects the whole catalog.
func loadMediaCatalog(read func() ([]byte, error)) (*mediaCatalog, error) {
	data, err := read()
	if err != nil {
		return nil, fmt.Errorf("read media catalog: %w", err)
	}
	var cat mediaCatalog
	if err := json.Unmarshal(data, &cat); err != nil {
		return nil, fmt.Errorf("parse media catalog: %w", err)
	}
	if err := validateMediaCatalog(&cat); err != nil {
		return nil, err
	}
	return &cat, nil
}

// validateMediaCatalog enforces the referential integrity rules. Pure
// function so tests can drive it with hand-built catalogs.
func validateMediaCatalog(cat *mediaCatalog) error {
	if cat == nil {
		return fmt.Errorf("media catalog is nil")
	}
	if cat.SchemaVersion != mediaCatalogSchemaVersion {
		return fmt.Errorf("media catalog schema version %d not supported (want %d)", cat.SchemaVersion, mediaCatalogSchemaVersion)
	}
	seenModels := map[string]bool{}
	for mi := range cat.Models {
		m := &cat.Models[mi]
		if m.ID == "" {
			return fmt.Errorf("model #%d: empty id", mi)
		}
		if seenModels[m.ID] {
			return fmt.Errorf("model %s: duplicate id", m.ID)
		}
		seenModels[m.ID] = true
		if len(m.Variants) == 0 {
			return fmt.Errorf("model %s: no variants", m.ID)
		}
		seenVariants := map[string]bool{}
		for vi := range m.Variants {
			v := &m.Variants[vi]
			if v.ID == "" {
				return fmt.Errorf("model %s variant #%d: empty id", m.ID, vi)
			}
			if seenVariants[v.ID] {
				return fmt.Errorf("model %s: duplicate variant id %s", m.ID, v.ID)
			}
			seenVariants[v.ID] = true
			if err := validateMediaVariant(m.ID, v); err != nil {
				return err
			}
		}
	}
	return nil
}

// validateMediaVariant checks one variant: profile kind, component roles and
// per-component identity fields. Components must be unique by identity.
func validateMediaVariant(modelID string, v *MediaVariant) error {
	switch v.Profile.Kind {
	case "image":
		// supported in P1
	case "video":
		return fmt.Errorf("model %s variant %s: video profiles are not open yet (plan §7 P1)", modelID, v.ID)
	default:
		return fmt.Errorf("model %s variant %s: unknown profile kind %q", modelID, v.ID, v.Profile.Kind)
	}
	if v.Profile.Family == "" {
		return fmt.Errorf("model %s variant %s: empty profile family", modelID, v.ID)
	}
	if len(v.Components) == 0 {
		return fmt.Errorf("model %s variant %s: no components", modelID, v.ID)
	}
	seen := map[string]bool{}
	for ci := range v.Components {
		c := &v.Components[ci]
		if !mediaKnownRoles[c.Role] {
			return fmt.Errorf("model %s variant %s: unknown component role %q", modelID, v.ID, c.Role)
		}
		if c.Repo == "" || c.Revision == "" || c.File == "" {
			return fmt.Errorf("model %s variant %s: component %q missing repo/revision/file", modelID, v.ID, c.Role)
		}
		if c.Size < 0 {
			return fmt.Errorf("model %s variant %s: component %q negative size", modelID, v.ID, c.Role)
		}
		if c.SHA256 != "" && len(c.SHA256) != 64 {
			return fmt.Errorf("model %s variant %s: component %q sha256 must be 64 hex chars", modelID, v.ID, c.Role)
		}
		if seen[c.Identity()] {
			return fmt.Errorf("model %s variant %s: duplicate component identity %s", modelID, v.ID, c.Identity())
		}
		seen[c.Identity()] = true
	}
	return nil
}

// MediaCatalogModel looks up a model by id (nil when absent).
func MediaCatalogModel(id string) (*MediaModel, error) {
	cat, err := GetMediaCatalog()
	if err != nil {
		return nil, err
	}
	for i := range cat.Models {
		if cat.Models[i].ID == id {
			return &cat.Models[i], nil
		}
	}
	return nil, nil
}

// MediaCatalogVariant looks up a variant by (modelID, variantID).
func MediaCatalogVariant(modelID, variantID string) (*MediaModel, *MediaVariant, error) {
	m, err := MediaCatalogModel(modelID)
	if err != nil {
		return nil, nil, err
	}
	if m == nil {
		return nil, nil, fmt.Errorf("media model %q not in catalog", modelID)
	}
	for i := range m.Variants {
		if m.Variants[i].ID == variantID {
			return m, &m.Variants[i], nil
		}
	}
	return nil, nil, fmt.Errorf("media model %q has no variant %q", modelID, variantID)
}

// MediaVariantUniqueComponents returns the variant's components deduped by
// asset identity (plan §5.3: the same asset referenced by two roles is
// downloaded/verified once). Order is stable (first occurrence wins).
func MediaVariantUniqueComponents(v *MediaVariant) []MediaComponent {
	out := make([]MediaComponent, 0, len(v.Components))
	seen := map[string]bool{}
	for _, c := range v.Components {
		if seen[c.Identity()] {
			continue
		}
		seen[c.Identity()] = true
		out = append(out, c)
	}
	return out
}

// MediaVariantTotalBytes sums the unique components' known sizes. Components
// with unknown size (Size==0) make the total partial: the second return is
// false and callers must surface "pending confirmation" instead of a number
// (plan §4.2: the whole package size shows as "pending confirmation").
func MediaVariantTotalBytes(v *MediaVariant) (int64, bool) {
	var total int64
	complete := true
	for _, c := range MediaVariantUniqueComponents(v) {
		if c.Size == 0 {
			complete = false
			continue
		}
		total += c.Size
	}
	return total, complete
}
