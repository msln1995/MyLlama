package core

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mediaModelsTestEnv pins the installs state file into a temp dir and swaps
// the variant lookup to a controlled two-component variant (tiny files). It
// returns a cleanup func and the catalog components used.
func mediaModelsTestEnv(t *testing.T) (cleanup func(), comps []MediaComponent) {
	t.Helper()
	dir := t.TempDir()
	oldPath := mediaInstallsPath
	mediaInstallsPath = func() string { return filepath.Join(dir, "installs.json") }
	oldLookup := mediaVariantLookup
	comps = []MediaComponent{
		{Role: MediaRoleDiffusion, Repo: "test/repo", Revision: "rev1", File: "diffusion.bin", Size: 0, SHA256: "", Format: "gguf"},
	}
	// Build the controlled components from real files so hashes are pinned.
	f1 := filepath.Join(dir, "src", "diffusion.bin")
	if err := os.MkdirAll(filepath.Dir(f1), 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(f1, []byte("DIFFUSION-WEIGHTS-CONTENT"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	f2 := filepath.Join(dir, "src", "te.bin")
	if err := os.WriteFile(f2, []byte("TE-CONTENT"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	sum1, err := fileSHA256(f1)
	if err != nil {
		t.Fatalf("sha: %v", err)
	}
	sum2, err := fileSHA256(f2)
	if err != nil {
		t.Fatalf("sha: %v", err)
	}
	comps = []MediaComponent{
		{Role: MediaRoleDiffusion, Repo: "test/repo", Revision: "rev1", File: "diffusion.bin", Size: 25, SHA256: sum1, Format: "gguf"},
		{Role: MediaRoleTextEncoder, Repo: "test/repo2", Revision: "rev1", File: "te.bin", Size: 10, SHA256: sum2, Format: "gguf"},
	}
	mediaVariantLookup = func(modelID, variantID string) (*MediaModel, *MediaVariant, error) {
		if modelID != "m" || variantID != "v" {
			return nil, nil, os.ErrNotExist
		}
		return &MediaModel{ID: "m"}, &MediaVariant{
			ID: "v", Label: "test", Components: comps,
			Profile: MediaRunnerProfile{Family: "z-image-turbo", Kind: "image"},
		}, nil
	}
	return func() {
		mediaInstallsPath = oldPath
		mediaVariantLookup = oldLookup
	}, comps
}

func TestVerifyFileIdentity(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f.bin")
	if err := os.WriteFile(p, []byte("hello-media"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	sum, _ := fileSHA256(p)
	if err := verifyFileIdentity(p, int64(len("hello-media")), sum); err != nil {
		t.Fatalf("identity verify should pass: %v", err)
	}
	if err := verifyFileIdentity(p, 999, sum); err == nil || !strings.Contains(err.Error(), "size") {
		t.Fatalf("want size mismatch error, got %v", err)
	}
	if err := verifyFileIdentity(p, int64(len("hello-media")), strings.Repeat("0", 64)); err == nil || !strings.Contains(err.Error(), "mismatch") {
		t.Fatalf("want sha mismatch error, got %v", err)
	}
	if err := verifyFileIdentity(p, int64(len("hello-media")), ""); err == nil {
		t.Fatalf("unpinned component must never verify")
	}
}

func TestMediaMatchFilesToComponents(t *testing.T) {
	cleanup, comps := mediaModelsTestEnv(t)
	defer cleanup()
	dir := t.TempDir()
	good := filepath.Join(dir, "diffusion.bin")
	if err := os.WriteFile(good, []byte("DIFFUSION-WEIGHTS-CONTENT"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	// Same bytes under a misleading name: content identity still matches.
	twin := filepath.Join(dir, "random-name.bin")
	if err := os.WriteFile(twin, []byte("TE-CONTENT"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	// Right name, wrong bytes: name alone must only produce a hint.
	impostor := filepath.Join(dir, "diffusion.bin.bak")
	if err := os.WriteFile(impostor, []byte("SOMETHING-ELSE"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	matches, err := mediaMatchFilesToComponents([]string{good, twin, impostor}, comps)
	if err != nil {
		t.Fatalf("match: %v", err)
	}
	if matches[0].Matched == nil || matches[0].Matched.Role != MediaRoleDiffusion {
		t.Fatalf("good file must match diffusion by content, got %+v", matches[0])
	}
	if matches[1].Matched == nil || matches[1].Matched.Role != MediaRoleTextEncoder {
		t.Fatalf("content match must ignore file name, got %+v", matches[1])
	}
	if matches[2].Matched != nil {
		t.Fatalf("impostor must not match")
	}
	if !strings.Contains(matches[2].Hint, "content identity differs") {
		t.Fatalf("impostor hint = %q, want content-identity hint", matches[2].Hint)
	}
}

func TestMediaImportVariantLifecycle(t *testing.T) {
	cleanup, _ := mediaModelsTestEnv(t)
	defer cleanup()
	dir := t.TempDir()
	good := filepath.Join(dir, "diffusion.bin")
	te := filepath.Join(dir, "te.bin")
	if err := os.WriteFile(good, []byte("DIFFUSION-WEIGHTS-CONTENT"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := os.WriteFile(te, []byte("TE-CONTENT"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Missing role rejects the import.
	if _, err := MediaImportVariant("m", "v", map[string]string{MediaRoleDiffusion: good}); err == nil || !strings.Contains(err.Error(), "one path per role") {
		t.Fatalf("want role-count rejection, got %v", err)
	}
	// Wrong bytes for a role reject the import.
	if _, err := MediaImportVariant("m", "v", map[string]string{
		MediaRoleDiffusion: te, MediaRoleTextEncoder: te,
	}); err == nil || !strings.Contains(err.Error(), "failed verification") {
		t.Fatalf("want verification failure, got %v", err)
	}
	// Happy path: both roles verified, status ready.
	rec, err := MediaImportVariant("m", "v", map[string]string{
		MediaRoleDiffusion: good, MediaRoleTextEncoder: te,
	})
	if err != nil {
		t.Fatalf("import: %v", err)
	}
	if len(rec.Components) != 2 {
		t.Fatalf("record components = %d, want 2", len(rec.Components))
	}
	for _, c := range rec.Components {
		if c.Kind != "external" {
			t.Fatalf("P1 imports are external refs, got kind %q", c.Kind)
		}
	}
	st, comps, err := MediaInstallStatus("m", "v")
	if err != nil || st != "ready" || len(comps) != 2 {
		t.Fatalf("status = %s/%v/%v, want ready with 2 components", st, comps, err)
	}

	// Tampering with an external file flips the status to invalid.
	if err := os.WriteFile(te, []byte("TAMPERED"), 0644); err != nil {
		t.Fatalf("tamper: %v", err)
	}
	st, _, err = MediaInstallStatus("m", "v")
	if err != nil || st != "invalid" {
		t.Fatalf("status after tamper = %s/%v, want invalid", st, err)
	}

	// Removing the install only drops the reference; files stay on disk.
	if err := MediaRemoveInstall("m", "v"); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if _, err := os.Stat(te); err != nil {
		t.Fatalf("external file must never be deleted, got %v", err)
	}
	if err := MediaRemoveInstall("m", "v"); err == nil {
		t.Fatalf("second remove must report no record")
	}
	st, _, err = MediaInstallStatus("m", "v")
	if err != nil || st != "not_installed" {
		t.Fatalf("status after remove = %s/%v, want not_installed", st, err)
	}
}
