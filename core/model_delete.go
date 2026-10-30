package core

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ─── Model deletion ──────────────────────────────────────────────
// Safe deletion of scanned models from disk. Ensures path containment
// inside registered model scan directories (directory traversal defense),
// guards against deleting running or loaded models, removes empty parent
// directories, and cleans up model configs, download tasks, and cache.

// DeleteModel deletes a model from disk by its main file path.
//
// Safety and Security Rules:
// 1. Path must not be empty and must be valid.
// 2. Traversal defense: path must be strictly contained within one of modelScanDirs().
// 3. File existence check: must be an existing file.
// 4. Runtime defense: cannot delete a model that is currently running or loaded.
func DeleteModel(modelPath string) error {
	trimmed := strings.TrimSpace(modelPath)
	if trimmed == "" {
		return errors.New(tr("模型路径不能为空", "model path cannot be empty"))
	}

	cleanPath := filepath.Clean(trimmed)
	absPath, err := filepath.Abs(cleanPath)
	if err != nil {
		return fmt.Errorf("%s: %w", tr("无效的模型路径", "invalid model path"), err)
	}

	// 1. Traversal / boundary defense: ensure absPath is strictly inside one of the scanned model roots
	roots := modelScanDirs()
	var matchedRoot string
	for _, root := range roots {
		absRoot, err := filepath.Abs(filepath.Clean(root))
		if err != nil {
			continue
		}
		rel, err := filepath.Rel(absRoot, absPath)
		if err == nil && !strings.HasPrefix(rel, "..") && rel != "." {
			matchedRoot = absRoot
			break
		}
	}
	if matchedRoot == "" {
		return errors.New(tr("安全错误：模型路径不在允许的模型目录内", "security error: model path is outside allowed model directories"))
	}

	// 2. File stat check
	fi, err := os.Stat(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf(tr("未找到模型文件：%s", "model file not found: %s"), absPath)
		}
		return err
	}
	if fi.IsDir() {
		return errors.New(tr("模型路径必须指向模型文件而非目录", "model path must point to a file, not a directory"))
	}

	// 3. Compute relative layout and locate target model
	rel, err := filepath.Rel(matchedRoot, absPath)
	if err != nil {
		return err
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")

	models := scanModels()
	var target *ModelInfo
	for i := range models {
		if fiM, err := os.Stat(models[i].Path); err == nil && os.SameFile(fi, fiM) {
			target = &models[i]
			break
		}
	}
	// In 3-level layout (author/variant/files...), match any model residing in the same variant directory
	if target == nil && len(parts) >= 3 {
		variantDir := filepath.Join(matchedRoot, parts[0], parts[1])
		for i := range models {
			cleanMPath := filepath.Clean(models[i].Path)
			if strings.HasPrefix(cleanMPath, variantDir+string(filepath.Separator)) || cleanMPath == variantDir {
				target = &models[i]
				break
			}
		}
	}

	// 4. Runtime checks: prevent deleting running or loaded models
	serverMu.Lock()
	running := serverRunning
	serving := directServingModel
	port := serverPort
	serverMu.Unlock()

	if running {
		if platformGOOS == "android" {
			if target != nil && serving == target.Name {
				return fmt.Errorf(tr("模型 %q 正在运行中，请先停止服务后再删除", "model %q is currently running; please stop the service before deleting"), target.Name)
			}
		} else if port > 0 {
			if loaded, err := fetchRouterModels(port); err == nil {
				for _, lm := range loaded {
					if target != nil && (lm.ID == target.Alias || lm.ID == target.Name) {
						return fmt.Errorf(tr("模型 %q 当前已加载至内存中，请先卸载后再删除", "model %q is currently loaded; please unload it before deleting"), target.Name)
					}
				}
			}
		}
	}

	// 5. Disk removal: determine variant directory vs loose file
	var deletedDir string
	if len(parts) >= 3 {
		// Three-level layout: matchedRoot / parts[0] (author) / parts[1] (variant) / ...
		variantDir := filepath.Join(matchedRoot, parts[0], parts[1])
		deletedDir = variantDir
		if err := os.RemoveAll(variantDir); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf(tr("删除模型目录失败：%w", "failed to remove model directory: %w"), err)
		}

		// Clean up author directory if empty
		authorDir := filepath.Join(matchedRoot, parts[0])
		if entries, err := os.ReadDir(authorDir); err == nil && len(entries) == 0 {
			_ = os.Remove(authorDir)
		}
	} else if len(parts) == 2 {
		// Two-level layout: matchedRoot / parts[0] (author) / parts[1] (file.gguf)
		if err := os.Remove(absPath); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf(tr("删除模型文件失败：%w", "failed to remove model file: %w"), err)
		}
		authorDir := filepath.Join(matchedRoot, parts[0])
		if entries, err := os.ReadDir(authorDir); err == nil && len(entries) == 0 {
			_ = os.Remove(authorDir)
		}
	} else {
		// Direct root file: matchedRoot / file.gguf
		if err := os.Remove(absPath); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf(tr("删除模型文件失败：%w", "failed to remove model file: %w"), err)
		}
	}

	// 6. Cleanup model config
	if target != nil {
		modelConfigsMu.Lock()
		delete(cachedModelConfigs, target.Name)
		modelConfigsMu.Unlock()
		saveConfig()
	}

	// 7. Cleanup download tasks
	if deletedDir != "" {
		cleanDlTasksForDir(deletedDir)
	} else {
		cleanDlTasksForFile(absPath)
	}

	// 8. Invalidate model cache
	invalidateModelCache()

	return nil
}

// cleanDlTasksForDir removes tasks whose DestDir matches or is inside dir.
func cleanDlTasksForDir(dir string) {
	dlTasksMu.Lock()
	var remaining []*DlTask
	var modified bool
	absDir, err2 := filepath.Abs(filepath.Clean(dir))
	for _, t := range dlTasks {
		absDest, err := filepath.Abs(filepath.Clean(t.DestDir))
		if err == nil && err2 == nil && (absDest == absDir || strings.HasPrefix(absDest, absDir+string(filepath.Separator))) {
			if t.cancel != nil {
				t.cancel()
			}
			modified = true
			continue
		}
		remaining = append(remaining, t)
	}
	if modified {
		dlTasks = remaining
	}
	dlTasksMu.Unlock()
	if modified {
		persistTasksNow()
	}
}

// cleanDlTasksForFile removes tasks whose destination file matches filePath.
func cleanDlTasksForFile(filePath string) {
	dlTasksMu.Lock()
	var remaining []*DlTask
	var modified bool
	absTarget, errTarget := filepath.Abs(filepath.Clean(filePath))
	for _, t := range dlTasks {
		taskFile := filepath.Join(t.DestDir, t.FileName)
		absTaskFile, err := filepath.Abs(filepath.Clean(taskFile))
		if err == nil && errTarget == nil && absTaskFile == absTarget {
			if t.cancel != nil {
				t.cancel()
			}
			modified = true
			continue
		}
		remaining = append(remaining, t)
	}
	if modified {
		dlTasks = remaining
	}
	dlTasksMu.Unlock()
	if modified {
		persistTasksNow()
	}
}
