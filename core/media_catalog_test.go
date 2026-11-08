package core

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestMediaCatalogEmbeddedLoad verifies the shipped catalog parses, passes
// validation and carries the P0-verified Z-Image Turbo package with its
// three sha256-anchored components.
func TestMediaCatalogEmbeddedLoad(t *testing.T) {
	cat, err := GetMediaCatalog()
	if err != nil {
		t.Fatalf("GetMediaCatalog: %v", err)
	}
	if cat.SchemaVersion != mediaCatalogSchemaVersion {
		t.Fatalf("schema version = %d, want %d", cat.SchemaVersion, mediaCatalogSchemaVersion)
	}
	m, err := MediaCatalogModel("z-image-turbo")
	if err != nil || m == nil {
		t.Fatalf("z-image-turbo missing from catalog: %v", err)
	}
	if len(m.Variants) != 1 {
		t.Fatalf("z-image-turbo variants = %d, want 1", len(m.Variants))
	}
	v := &m.Variants[0]
	comps := MediaVariantUniqueComponents(v)
	if len(comps) != 3 {
		t.Fatalf("unique components = %d, want 3 (diffusion/text_encoder/vae)", len(comps))
	}
	roles := map[string]bool{}
	for _, c := range comps {
		roles[c.Role] = true
		if c.Size <= 0 || len(c.SHA256) != 64 {
			t.Fatalf("component role %s not fully pinned (size=%d sha len=%d)", c.Role, c.Size, len(c.SHA256))
		}
	}
	for _, want := range []string{MediaRoleDiffusion, MediaRoleTextEncoder, MediaRoleVAE} {
		if !roles[want] {
			t.Fatalf("variant missing role %s", want)
		}
	}
	total, complete := MediaVariantTotalBytes(v)
	if !complete {
		t.Fatalf("variant total bytes incomplete: %d", total)
	}
	// The P0 measured sum of the three pinned components.
	if total != 3864250304+2497281120+335304388 {
		t.Fatalf("total bytes = %d, want the P0 pinned sum", total)
	}
	// The profile is the fixed image profile of P1.
	if v.Profile.Kind != "image" || v.Profile.Family != "z-image-turbo" {
		t.Fatalf("profile = %s/%s, want image/z-image-turbo", v.Profile.Kind, v.Profile.Family)
	}
}

// buildCatalog is a test helper producing a catalog JSON with overrides
// applied to the valid baseline.
func buildCatalog(t *testing.T, mutate func(m *MediaModel)) string {
	t.Helper()
	cat := &mediaCatalog{
		SchemaVersion: mediaCatalogSchemaVersion,
		Models: []MediaModel{
			{
				ID: "m1", Family: "f", Name: "M1", Developer: "d", License: "l",
				Variants: []MediaVariant{
					{
						ID: "v1", Label: "L",
						Components: []MediaComponent{
							{Role: MediaRoleDiffusion, Repo: "r", Revision: "rev", File: "a.gguf", Size: 10, SHA256: strings.Repeat("a", 64)},
							{Role: MediaRoleTextEncoder, Repo: "r2", Revision: "rev", File: "b.gguf", Size: 0},
						},
						Profile: MediaRunnerProfile{Family: "z-image-turbo", Kind: "image", CFGScale: 1, Steps: 8, Sampler: "euler", MinSize: 256, MaxSize: 1024, SizeMultiple: 16},
					},
				},
			},
		},
	}
	if mutate != nil {
		mutate(&cat.Models[0])
	}
	data, err := json.Marshal(cat)
	if err != nil {
		t.Fatalf("marshal test catalog: %v", err)
	}
	return string(data)
}

func loadFromJSON(t *testing.T, s string) (*mediaCatalog, error) {
	t.Helper()
	return loadMediaCatalog(func() ([]byte, error) { return []byte(s), nil })
}

func TestMediaCatalogValidationRejections(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(m *MediaModel)
		wantErr string
	}{
		{"wrong schema version", nil, ""},
		{"unknown role", func(m *MediaModel) { m.Variants[0].Components[0].Role = "wat" }, "unknown component role"},
		{"video profile not open", func(m *MediaModel) { m.Variants[0].Profile.Kind = "video" }, "video profiles are not open"},
		{"unknown profile kind", func(m *MediaModel) { m.Variants[0].Profile.Kind = "audio" }, "unknown profile kind"},
		{"empty profile family", func(m *MediaModel) { m.Variants[0].Profile.Family = "" }, "empty profile family"},
		{"duplicate variant id", func(m *MediaModel) {
			m.Variants = append(m.Variants, m.Variants[0])
		}, "duplicate variant id"},
		{"no variants", func(m *MediaModel) { m.Variants = nil }, "no variants"},
		{"missing component file", func(m *MediaModel) { m.Variants[0].Components[0].File = "" }, "missing repo/revision/file"},
		{"bad sha256 length", func(m *MediaModel) { m.Variants[0].Components[0].SHA256 = "abc" }, "sha256 must be 64 hex chars"},
		{"negative size", func(m *MediaModel) { m.Variants[0].Components[0].Size = -1 }, "negative size"},
		{"duplicate component identity", func(m *MediaModel) {
			m.Variants[0].Components = append(m.Variants[0].Components, m.Variants[0].Components[0])
		}, "duplicate component identity"},
		{"no components", func(m *MediaModel) { m.Variants[0].Components = nil }, "no components"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.mutate == nil {
				// wrong schema version: patch the JSON directly.
				s := buildCatalog(t, nil)
				s = strings.Replace(s, `"schemaVersion":1`, `"schemaVersion":99`, 1)
				if _, err := loadFromJSON(t, s); err == nil || !strings.Contains(err.Error(), "schema version") {
					t.Fatalf("want schema version error, got %v", err)
				}
				return
			}
			s := buildCatalog(t, tc.mutate)
			_, err := loadFromJSON(t, s)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

// TestMediaVariantTotalBytesUnknownSize covers the "pending confirmation"
// rule: a component with unknown size makes the total partial, and the
// partial sum still counts every known component exactly once (dedupe by
// identity).
func TestMediaVariantTotalBytesUnknownSize(t *testing.T) {
	v := &MediaVariant{
		ID: "v",
		Components: []MediaComponent{
			{Role: MediaRoleDiffusion, Repo: "r", Revision: "x", File: "same.bin", Size: 100},
			{Role: MediaRoleVAE, Repo: "r", Revision: "x", File: "same.bin", Size: 100}, // same identity: counted once
			{Role: MediaRoleTextEncoder, Repo: "r2", Revision: "x", File: "te.bin", Size: 0},
		},
	}
	total, complete := MediaVariantTotalBytes(v)
	if complete {
		t.Fatalf("total must be partial when a size is unknown")
	}
	if total != 100 {
		t.Fatalf("total = %d, want 100 (duplicate identity counted once)", total)
	}
}
