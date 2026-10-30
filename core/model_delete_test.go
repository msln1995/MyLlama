package core

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestDeleteModelValidation(t *testing.T) {
	// 1. Empty path
	if err := DeleteModel(""); err == nil {
		t.Fatalf("DeleteModel(\"\") want error, got nil")
	}

	// 2. Traversal attempt outside modelScanDirs
	tempDir := t.TempDir()
	outsideFile := filepath.Join(tempDir, "outside.gguf")
	if err := os.WriteFile(outsideFile, []byte("fake"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := DeleteModel(outsideFile); err == nil {
		t.Fatalf("DeleteModel(outsideFile) want security error, got nil")
	}

	// 3. Directory instead of file
	modelsDir := t.TempDir()
	origDefaultDir := defaultModelsDir
	defaultModelsDir = func() string { return modelsDir }
	defer func() { defaultModelsDir = origDefaultDir }()

	authorDir := filepath.Join(modelsDir, "testauthor")
	if err := os.MkdirAll(authorDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := DeleteModel(authorDir); err == nil {
		t.Fatalf("DeleteModel(dir) want error, got nil")
	}

	// 4. Non-existent file
	nonExistent := filepath.Join(authorDir, "missing.gguf")
	if err := DeleteModel(nonExistent); err == nil {
		t.Fatalf("DeleteModel(nonExistent) want error, got nil")
	}
}

func TestDeleteModelThreeLevelLayout(t *testing.T) {
	modelsDir := t.TempDir()
	origDefaultDir := defaultModelsDir
	defaultModelsDir = func() string { return modelsDir }
	defer func() { defaultModelsDir = origDefaultDir }()

	// Create 3-level layout: modelsDir / author / variant / file.gguf + extra shard
	variantDir := filepath.Join(modelsDir, "theauthor", "thevariant")
	if err := os.MkdirAll(variantDir, 0755); err != nil {
		t.Fatal(err)
	}
	mainFile := filepath.Join(variantDir, "model.gguf")
	shardFile := filepath.Join(variantDir, "model-00001-of-00002.gguf")

	// Write mock GGUF files with magic header
	magic := []byte("GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
	if err := os.WriteFile(mainFile, magic, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(shardFile, magic, 0644); err != nil {
		t.Fatal(err)
	}

	// Set cached config for this model
	modelConfigsMu.Lock()
	cachedModelConfigs["thevariant"] = ModelConfig{CtxSize: 4096}
	modelConfigsMu.Unlock()

	// Add a dlTask for this variantDir
	dlTasksMu.Lock()
	ctx, cancel := context.WithCancel(context.Background())
	dlTasks = []*DlTask{
		{
			ID:       "task-delete-test",
			ModelID:  "theauthor/thevariant",
			DestDir:  variantDir,
			FileName: "model.gguf",
			Status:   "done",
			ctx:      ctx,
			cancel:   cancel,
		},
	}
	dlTasksMu.Unlock()

	// Invalidate and verify it's scanned
	invalidateModelCache()
	scanned := scanModels()
	if len(scanned) == 0 {
		t.Fatalf("scanModels() did not find the model")
	}
	t.Logf("scanned model name=%q, path=%q", scanned[0].Name, scanned[0].Path)

	// Delete model
	if err := DeleteModel(mainFile); err != nil {
		t.Fatalf("DeleteModel failed: %v", err)
	}

	// Verify variant directory is gone
	if _, err := os.Stat(variantDir); !os.IsNotExist(err) {
		t.Errorf("variantDir %s still exists after DeleteModel", variantDir)
	}

	// Verify author directory is cleaned up because it was empty
	authorDir := filepath.Join(modelsDir, "theauthor")
	if _, err := os.Stat(authorDir); !os.IsNotExist(err) {
		t.Errorf("authorDir %s still exists after DeleteModel", authorDir)
	}

	// Verify cachedModelConfigs entry was removed
	modelConfigsMu.Lock()
	_, cfgExists := cachedModelConfigs["thevariant"]
	modelConfigsMu.Unlock()
	if cfgExists {
		t.Errorf("cachedModelConfigs['thevariant'] still exists after DeleteModel")
	}

	// Verify dlTasks entry was cleaned up
	dlTasksMu.Lock()
	taskCount := len(dlTasks)
	dlTasksMu.Unlock()
	if taskCount != 0 {
		t.Errorf("dlTasks still has %d tasks, want 0", taskCount)
	}
}

func TestDeleteModelTwoLevelLayout(t *testing.T) {
	modelsDir := t.TempDir()
	origDefaultDir := defaultModelsDir
	defaultModelsDir = func() string { return modelsDir }
	defer func() { defaultModelsDir = origDefaultDir }()

	// Create 2-level layout: modelsDir / author / file1.gguf & file2.gguf
	authorDir := filepath.Join(modelsDir, "theauthor")
	if err := os.MkdirAll(authorDir, 0755); err != nil {
		t.Fatal(err)
	}
	file1 := filepath.Join(authorDir, "model1.gguf")
	file2 := filepath.Join(authorDir, "model2.gguf")

	magic := []byte("GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
	if err := os.WriteFile(file1, magic, 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file2, magic, 0644); err != nil {
		t.Fatal(err)
	}

	invalidateModelCache()

	// Delete file1: authorDir must NOT be deleted because file2 still exists
	if err := DeleteModel(file1); err != nil {
		t.Fatalf("DeleteModel(file1) failed: %v", err)
	}

	if _, err := os.Stat(file1); !os.IsNotExist(err) {
		t.Errorf("file1 still exists after DeleteModel")
	}
	if _, err := os.Stat(authorDir); os.IsNotExist(err) {
		t.Errorf("authorDir was deleted while file2 was still present")
	}

	// Delete file2: now authorDir is empty, should be removed
	if err := DeleteModel(file2); err != nil {
		t.Fatalf("DeleteModel(file2) failed: %v", err)
	}
	if _, err := os.Stat(file2); !os.IsNotExist(err) {
		t.Errorf("file2 still exists after DeleteModel")
	}
	if _, err := os.Stat(authorDir); !os.IsNotExist(err) {
		t.Errorf("authorDir still exists when empty")
	}
}

func TestDeleteModelRejectsRunning(t *testing.T) {
	modelsDir := t.TempDir()
	origDefaultDir := defaultModelsDir
	defaultModelsDir = func() string { return modelsDir }
	defer func() { defaultModelsDir = origDefaultDir }()

	variantDir := filepath.Join(modelsDir, "author", "running-model")
	if err := os.MkdirAll(variantDir, 0755); err != nil {
		t.Fatal(err)
	}
	mainFile := filepath.Join(variantDir, "model.gguf")
	magic := []byte("GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
	if err := os.WriteFile(mainFile, magic, 0644); err != nil {
		t.Fatal(err)
	}

	invalidateModelCache()

	// Simulate Android direct mode running
	origGOOS := platformGOOS
	platformGOOS = "android"
	defer func() { platformGOOS = origGOOS }()

	serverMu.Lock()
	serverRunning = true
	directServingModel = "running-model"
	serverMu.Unlock()
	defer func() {
		serverMu.Lock()
		serverRunning = false
		directServingModel = ""
		serverMu.Unlock()
	}()

	err := DeleteModel(mainFile)
	if err == nil {
		t.Fatalf("DeleteModel want error when model is running, got nil")
	}
	_ = runtime.GOOS
}
