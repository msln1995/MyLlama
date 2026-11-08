// Media package install state and local import matching. The catalog (see
// media_catalog.go) defines what a variant needs; this file tracks which
// components actually exist locally and verifies them by content.
//
// Two component sources exist (plan §5.3):
//   - "external": a user-owned file referenced in place. External files are
//     never moved, copied or deleted by the app; their identity (size +
//     sha256) is recorded and re-verified before each use.
//   - "managed": an app-managed copy inside the content-addressed store
//     (P2 install flow writes these; P1 only reads the layout).
//
// Readiness is anchored to verified bytes, not file names: an install is
// "ready" only while every recorded component still matches its pinned
// identity.
package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// mediaInstallsSchemaVersion guards the installs state file layout.
const mediaInstallsSchemaVersion = 1

// MediaComponentSource records one resolved component of an installed
// variant: which catalog asset it stands for and where the bytes live.
type MediaComponentSource struct {
	Role     string `json:"role"`
	Identity string `json:"identity"` // Repo@Revision:File dedupe key
	Kind     string `json:"kind"`     // "external" | "managed"
	Path     string `json:"path"`     // absolute local path
	Size     int64  `json:"size"`
	SHA256   string `json:"sha256"`
}

// MediaInstallRecord is one installed (model, variant) with resolved
// component sources. External references coexist with managed copies; the
// record never implies ownership of external files.
type MediaInstallRecord struct {
	ModelID    string                 `json:"modelId"`
	VariantID  string                 `json:"variantId"`
	Components []MediaComponentSource `json:"components"`
}

// mediaInstallsFile is the persisted state document.
type mediaInstallsFile struct {
	SchemaVersion int                  `json:"schemaVersion"`
	Installs      []MediaInstallRecord `json:"installs"`
}

// mediaInstallsPath is the state-file path override (tests pin it; production
// resolves through mediaInstallsFilePath).
var mediaInstallsPath = mediaInstallsFilePath

var mediaInstallsMu sync.Mutex

// mediaVariantLookup is the seam tests swap to drive install flows with
// controlled tiny components (production: MediaCatalogVariant).
var mediaVariantLookup = MediaCatalogVariant

// loadMediaInstalls reads the installs state file. A missing file is an empty
// state; a corrupt file degrades to empty with a [WARN] (the catalog can
// re-verify anything, so no data is destroyed).
func loadMediaInstalls() *mediaInstallsFile {
	data, err := os.ReadFile(mediaInstallsPath())
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[WARN] Cannot read media installs file, starting empty: %v", err)
		}
		return &mediaInstallsFile{SchemaVersion: mediaInstallsSchemaVersion}
	}
	var f mediaInstallsFile
	if err := json.Unmarshal(data, &f); err != nil {
		log.Printf("[WARN] Cannot parse media installs file, starting empty: %v", err)
		return &mediaInstallsFile{SchemaVersion: mediaInstallsSchemaVersion}
	}
	if f.SchemaVersion != mediaInstallsSchemaVersion {
		log.Printf("[WARN] Media installs schema version %d unsupported, starting empty", f.SchemaVersion)
		return &mediaInstallsFile{SchemaVersion: mediaInstallsSchemaVersion}
	}
	return &f
}

// saveMediaInstalls atomically persists the installs state.
func saveMediaInstalls(f *mediaInstallsFile) error {
	f.SchemaVersion = mediaInstallsSchemaVersion
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal media installs: %w", err)
	}
	if err := atomicWriteFile(mediaInstallsPath(), data, 0644); err != nil {
		return fmt.Errorf("write media installs: %w", err)
	}
	return nil
}

// findMediaInstall returns the install record for (model, variant) or nil.
// Caller must hold mediaInstallsMu.
func findMediaInstall(f *mediaInstallsFile, modelID, variantID string) *MediaInstallRecord {
	for i := range f.Installs {
		if f.Installs[i].ModelID == modelID && f.Installs[i].VariantID == variantID {
			return &f.Installs[i]
		}
	}
	return nil
}

// verifyFileIdentity streams a file and checks its exact size and sha256
// against the expected values. A component with unknown size/hash (catalog
// "pending confirmation") cannot be verified and always fails here — readiness
// is never claimed on unpinned bytes.
func verifyFileIdentity(path string, wantSize int64, wantSHA string) error {
	fi, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("stat %s: %w", path, err)
	}
	if wantSize > 0 && fi.Size() != wantSize {
		return fmt.Errorf("%s: size %d, want %d", filepath.Base(path), fi.Size(), wantSize)
	}
	if wantSHA == "" {
		return fmt.Errorf("%s: no pinned sha256 to verify against", filepath.Base(path))
	}
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("hash %s: %w", path, err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != wantSHA {
		return fmt.Errorf("%s: sha256 mismatch (file changed or corrupted)", filepath.Base(path))
	}
	return nil
}

// MediaImportMatch is the per-file result of an import inspection (plan §5.3:
// import first reports identification, reusable files, missing pieces — it
// never silently installs by file name).
type MediaImportMatch struct {
	Path     string          `json:"path"`
	Size     int64           `json:"size"`
	SHA256   string          `json:"sha256"`
	Matched  *MediaComponent `json:"matched,omitempty"` // catalog component with identical content identity, if any
	FileName string          `json:"fileName"`
	Hint     string          `json:"hint,omitempty"` // candidate hint from the file name when no content match exists
}

// mediaMatchFilesToComponents hashes every path and matches content identity
// against the components. Pure in its inputs (no state), so tests can drive
// it with controlled files; the binding wrapper supplies catalog components.
func mediaMatchFilesToComponents(paths []string, components []MediaComponent) ([]MediaImportMatch, error) {
	out := make([]MediaImportMatch, 0, len(paths))
	for _, p := range paths {
		abs, err := filepath.Abs(p)
		if err != nil {
			return nil, fmt.Errorf("resolve %s: %w", p, err)
		}
		fi, err := os.Stat(abs)
		if err != nil {
			return nil, fmt.Errorf("stat %s: %w", p, err)
		}
		sum, err := fileSHA256(abs)
		if err != nil {
			return nil, err
		}
		m := MediaImportMatch{Path: abs, Size: fi.Size(), SHA256: sum, FileName: filepath.Base(abs)}
		for i := range components {
			c := components[i]
			if c.SHA256 != "" && c.SHA256 == sum && (c.Size == 0 || c.Size == fi.Size()) {
				cc := c
				m.Matched = &cc
				break
			}
		}
		if m.Matched == nil {
			// File name only ever produces a hint, never a match (plan §5.3
			// rule 4: 文件名只作候选，不能据文件名自动认定匹配).
			base := strings.ToLower(filepath.Base(abs))
			for i := range components {
				if strings.Contains(base, strings.ToLower(filepath.Base(components[i].File))) {
					m.Hint = "file name resembles catalog component " + components[i].File + ", but content identity differs"
					break
				}
			}
			if m.Hint == "" {
				m.Hint = "no catalog component matches this content; unknown files are not auto-installed"
			}
		}
		out = append(out, m)
	}
	return out, nil
}

// fileSHA256 streams a file through sha256 (4 MiB chunks).
func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("hash %s: %w", path, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// MediaImportInspect inspects candidate files against a variant's components
// and returns the match report plus the missing-component list (identified by
// role) the user still has to provide.
func MediaImportInspect(modelID, variantID string, paths []string) ([]MediaImportMatch, []string, error) {
	_, v, err := mediaVariantLookup(modelID, variantID)
	if err != nil {
		return nil, nil, err
	}
	matches, err := mediaMatchFilesToComponents(paths, v.Components)
	if err != nil {
		return nil, nil, err
	}
	matchedRoles := map[string]bool{}
	for _, m := range matches {
		if m.Matched != nil {
			matchedRoles[m.Matched.Role] = true
		}
	}
	var missing []string
	for _, c := range v.Components {
		if !matchedRoles[c.Role] {
			missing = append(missing, c.Role)
		}
	}
	return matches, missing, nil
}

// MediaImportVariant records an install whose components are external file
// references. Every role of the variant must be provided and every file must
// verify against its pinned identity (size + sha256) at import time; anything
// else rejects the import with the failing roles listed (plan §5.3: only
// imports matching the built-in catalog are supported).
func MediaImportVariant(modelID, variantID string, pathByRole map[string]string) (*MediaInstallRecord, error) {
	_, v, err := mediaVariantLookup(modelID, variantID)
	if err != nil {
		return nil, err
	}
	if len(pathByRole) != len(v.Components) {
		return nil, fmt.Errorf("import needs one path per role (%d), got %d", len(v.Components), len(pathByRole))
	}
	record := MediaInstallRecord{ModelID: modelID, VariantID: variantID}
	for _, c := range v.Components {
		p, ok := pathByRole[c.Role]
		if !ok || p == "" {
			return nil, fmt.Errorf("missing path for role %q", c.Role)
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			return nil, fmt.Errorf("resolve %s: %w", p, err)
		}
		if err := verifyFileIdentity(abs, c.Size, c.SHA256); err != nil {
			return nil, fmt.Errorf("component %q failed verification: %w", c.Role, err)
		}
		fi, _ := os.Stat(abs)
		record.Components = append(record.Components, MediaComponentSource{
			Role:     c.Role,
			Identity: c.Identity(),
			Kind:     "external",
			Path:     abs,
			Size:     fi.Size(),
			SHA256:   c.SHA256,
		})
	}
	mediaInstallsMu.Lock()
	defer mediaInstallsMu.Unlock()
	f := loadMediaInstalls()
	// One install per (model, variant): a re-import replaces the previous
	// record (same identity semantics as re-downloading a variant).
	replaced := false
	for i := range f.Installs {
		if f.Installs[i].ModelID == modelID && f.Installs[i].VariantID == variantID {
			f.Installs[i] = record
			replaced = true
		}
	}
	if !replaced {
		f.Installs = append(f.Installs, record)
	}
	if err := saveMediaInstalls(f); err != nil {
		return nil, err
	}
	log.Printf("[OK] Imported media variant %s/%s (%d external components)", modelID, variantID, len(record.Components))
	return &record, nil
}

// MediaInstallStatus returns the derived install status of a variant:
//   - "not_installed": no record
//   - "ready": record present and every component re-verifies right now
//   - "invalid": record present but at least one component fails verification
//     (file moved, modified or deleted) — the UI must surface which
//
// The status is always derived live: the state file never claims readiness on
// its own (plan §4.2 安装完整性 column).
func MediaInstallStatus(modelID, variantID string) (string, []MediaComponentSource, error) {
	if _, _, err := mediaVariantLookup(modelID, variantID); err != nil {
		return "", nil, err
	}
	mediaInstallsMu.Lock()
	f := loadMediaInstalls()
	rec := findMediaInstall(f, modelID, variantID)
	mediaInstallsMu.Unlock()
	if rec == nil {
		return "not_installed", nil, nil
	}
	status := "ready"
	for i := range rec.Components {
		src := &rec.Components[i]
		if err := verifyFileIdentity(src.Path, src.Size, src.SHA256); err != nil {
			log.Printf("[WARN] Media component %s/%s role %s no longer verifies: %v", modelID, variantID, src.Role, err)
			status = "invalid"
		}
	}
	return status, rec.Components, nil
}

// MediaListInstalls returns all install records with their live statuses.
func MediaListInstalls() ([]MediaInstallRecord, []string, error) {
	mediaInstallsMu.Lock()
	f := loadMediaInstalls()
	mediaInstallsMu.Unlock()
	statuses := make([]string, len(f.Installs))
	for i := range f.Installs {
		st, _, err := MediaInstallStatus(f.Installs[i].ModelID, f.Installs[i].VariantID)
		if err != nil {
			return nil, nil, err
		}
		statuses[i] = st
	}
	return f.Installs, statuses, nil
}

// MediaRemoveInstall deletes an install record. External files are only
// unreferenced, never touched on disk (plan §5.3 rule 6); managed copies
// (P2) will refcount before deletion.
func MediaRemoveInstall(modelID, variantID string) error {
	mediaInstallsMu.Lock()
	defer mediaInstallsMu.Unlock()
	f := loadMediaInstalls()
	kept := f.Installs[:0]
	for _, in := range f.Installs {
		if in.ModelID == modelID && in.VariantID == variantID {
			continue
		}
		kept = append(kept, in)
	}
	if len(kept) == len(f.Installs) {
		return fmt.Errorf("no install record for %s/%s", modelID, variantID)
	}
	f.Installs = kept
	return saveMediaInstalls(f)
}
