<?php
// Simple video upload & link import interface
// Stores metadata in videos.json and uploaded files in uploads/

// Configuration
$uploadDir = __DIR__ . '/uploads';
$dataFile = __DIR__ . '/videos.json';
// $maxFileSize = 200 * 1024 * 1024; // 200 MB (disabled - size check removed)
$allowedExt = ['mp4','webm','ogg','mov','mkv'];

// Ensure storage exists
if (!is_dir($uploadDir)) {
    mkdir($uploadDir, 0755, true);
}
if (!file_exists($dataFile)) {
    file_put_contents($dataFile, json_encode([]));
}

function loadVideos($file) {
    $json = @file_get_contents($file);
    $arr = json_decode($json, true);
    return is_array($arr) ? $arr : [];
}

function saveVideos($file, $arr) {
    $json = json_encode($arr, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    // atomic write
    file_put_contents($file . '.tmp', $json, LOCK_EX);
    rename($file . '.tmp', $file);
}

$message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $videos = loadVideos($dataFile);

    if (!empty($_POST['submit_type']) && $_POST['submit_type'] === 'upload') {
        if (!isset($_FILES['videoFile']) || $_FILES['videoFile']['error'] !== UPLOAD_ERR_OK) {
            $message = 'No file uploaded or upload error.';
        } else {
            $file = $_FILES['videoFile'];
            // Size limit check removed — we don't enforce a max file size here.
            // If you want to re-enable a maximum size, uncomment and set $maxFileSize above
            $ext = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
            if (!in_array($ext, $allowedExt)) {
                $message = 'Unsupported file type.';
            } else {
                $safe = preg_replace('/[^A-Za-z0-9._-]/', '_', pathinfo($file['name'], PATHINFO_FILENAME));
                $targetName = $safe . '_' . time() . '.' . $ext;
                $targetPath = $uploadDir . '/' . $targetName;
                if (move_uploaded_file($file['tmp_name'], $targetPath)) {
                    $entry = [
                        'type' => 'file',
                        'filename' => 'uploads/' . $targetName,
                        'originalName' => $file['name'],
                        'size' => $file['size'],
                        'uploaded_at' => date('c')
                    ];
                    $videos[] = $entry;
                    saveVideos($dataFile, $videos);
                    $message = 'File uploaded successfully.';
                } else {
                    $message = 'Failed to move uploaded file.';
                }
            }
        }
    } elseif (!empty($_POST['submit_type']) && $_POST['submit_type'] === 'link') {
        $url = trim($_POST['videoUrl'] ?? '');
        if (empty($url) || !filter_var($url, FILTER_VALIDATE_URL)) {
            $message = 'Please provide a valid URL.';
        } else {
            // Basic whitelist: allow http/https
            $entry = [
                'type' => 'link',
                'url' => $url,
                'added_at' => date('c')
            ];
            $videos[] = $entry;
            saveVideos($dataFile, $videos);
            $message = 'Video link saved.';
        }
    }
}

$videos = loadVideos($dataFile);
// Toggle display of the Saved Videos section. Set to false to disable the UI part.
$showSavedSection = false;
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Video Upload Interface</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
<div class="container">
    <h1>Video Upload / Import</h1>
    <?php if ($message): ?>
        <div class="message"><?php echo htmlspecialchars($message); ?></div>
    <?php endif; ?>

    <div class="card">
        <h2>Video Upload</h2>
        <div class="tabs">
            <button type="button" class="tab-btn active" data-tab="upload">Upload File</button>
            <button type="button" class="tab-btn" data-tab="link">Video Link</button>
        </div>

        <div class="tab-content active" id="upload">
            <!-- UI saving disabled: onsubmit prevented to avoid posting to server -->
            <form method="post" enctype="multipart/form-data" id="uploadForm" onsubmit="return false;">
                <input type="hidden" name="submit_type" value="upload">
                <div class="dropzone" id="dropzone">
                    <input type="file" name="videoFile" id="videoFile" accept="video/mp4,video/webm,video/ogg" required>
                    <div class="drop-inner">
                        <div class="upload-icon">⬆️</div>
                        <div class="drop-text">Click to upload or drag and drop</div>
                        <div class="drop-sub">MP4, WebM, or OGG (max. 100MB)</div>
                    </div>
                </div>

                <div class="preview" id="filePreview" aria-hidden="true"></div>

                <div class="row">
                    <!-- <button type="submit">Upload</button> -->
                    <button type="button" id="removeFileBtn" class="btn-remove" style="display:none;margin-left:8px">Remove</button>
                    <small>Allowed: <?php echo implode(', ', $allowedExt); ?></small>
                </div>
                <div style="margin-top:8px;color:#9ca3af;font-size:0.9rem">Note: saving is disabled in the UI (server-side logic unchanged).</div>
            </form>
        </div>

        <div class="tab-content" id="link">
            <!-- UI saving disabled: onsubmit prevented to avoid posting to server -->
            <form method="post" id="linkForm" onsubmit="return false;">
                <input type="hidden" name="submit_type" value="link">
                <input type="url" id="videoUrl" name="videoUrl" placeholder="https://example.com/video.mp4 or YouTube link" required>

                <div class="preview" id="linkPreview" aria-hidden="true"></div>

                <div class="row">
                    <!-- <button type="submit">Save Link</button> -->
                </div>
                <div style="margin-top:8px;color:#9ca3af;font-size:0.9rem">Note: saving is disabled in the UI (server-side logic unchanged).</div>
            </form>
        </div>
    </div>

    <?php if ($showSavedSection): ?>
    <div class="card">
        <h2>Saved Videos</h2>
        <?php if (empty($videos)): ?>
            <p>No videos yet.</p>
        <?php else: ?>
            <div class="grid">
            <?php foreach (array_reverse($videos) as $v): ?>
                <div class="video-item">
                    <?php if ($v['type'] === 'file'): ?>
                        <video controls preload="metadata" width="320">
                            <source src="<?php echo htmlspecialchars($v['filename']); ?>">
                            Your browser does not support HTML5 video.
                        </video>
                        <div class="meta"><?php echo htmlspecialchars($v['originalName']); ?> — <?php echo round($v['size']/1024/1024,2); ?> MB</div>
                    <?php else: ?>
                        <?php
                        $url = $v['url'];
                        // simple youtube detection
                        $embed = null;
                        if (preg_match('#(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|v/))([A-Za-z0-9_-]{6,})#', $url, $m)) {
                            $embed = 'https://www.youtube.com/embed/' . $m[1];
                        } elseif (strpos($url, 'vimeo.com') !== false && preg_match('#vimeo\.com/(\d+)#', $url, $m2)) {
                            $embed = 'https://player.vimeo.com/video/' . $m2[1];
                        }
                        ?>
                        <?php if ($embed): ?>
                            <iframe src="<?php echo htmlspecialchars($embed); ?>" frameborder="0" allowfullscreen width="320" height="180"></iframe>
                        <?php else: ?>
                            <video controls preload="metadata" width="320">
                                <source src="<?php echo htmlspecialchars($url); ?>">
                                If this doesn't play, it's likely an unsupported remote format.
                            </video>
                        <?php endif; ?>
                        <div class="meta"><?php echo htmlspecialchars($v['url']); ?></div>
                    <?php endif; ?>
                </div>
            <?php endforeach; ?>
            </div>
        <?php endif; ?>
    </div>
    <?php else: ?>
    <div class="card">
        <h2>Saved Videos</h2>
        <p style="color:#6b7280">Saved Videos display has been disabled.</p>
    </div>
    <?php endif; ?>

</div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function(){
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(function(btn){
        btn.addEventListener('click', function(){
            document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
            btn.classList.add('active');
            const tab = btn.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
            const el = document.getElementById(tab);
            if(el) el.classList.add('active');
        });
    });

    // Drag & drop + previews
    const drop = document.getElementById('dropzone');
    const fileInput = document.getElementById('videoFile');
    const filePreview = document.getElementById('filePreview');
    const removeFileBtn = document.getElementById('removeFileBtn');
    let currentObjectUrl = null;

    function clearFilePreview(){
        if(currentObjectUrl){ URL.revokeObjectURL(currentObjectUrl); currentObjectUrl = null; }
        if(filePreview) filePreview.innerHTML = '';
        if(removeFileBtn) removeFileBtn.style.display = 'none';
    }

    function renderFilePreview(file){
        clearFilePreview();
        if(!file) return;
        const type = file.type || '';
        if(type.startsWith('video/')){
            const v = document.createElement('video');
            v.controls = true;
            v.preload = 'metadata';
            v.style.maxWidth = '100%';
            const src = URL.createObjectURL(file);
            currentObjectUrl = src;
            v.src = src;
            const info = document.createElement('div');
            info.className = 'meta';
            info.textContent = file.name + ' — ' + (Math.round(file.size/1024/1024*100)/100) + ' MB';
            if(filePreview){ filePreview.appendChild(v); filePreview.appendChild(info); filePreview.setAttribute('aria-hidden','false'); }
            if(removeFileBtn) removeFileBtn.style.display = 'inline-block';
        } else {
            if(filePreview) filePreview.textContent = 'Selected file is not a supported video.';
        }
    }

    if (drop && fileInput) {
        ['dragenter','dragover'].forEach(function(evt){
            drop.addEventListener(evt, function(e){ e.preventDefault(); e.stopPropagation(); drop.classList.add('dragover'); });
        });
        ['dragleave','drop'].forEach(function(evt){
            drop.addEventListener(evt, function(e){ e.preventDefault(); e.stopPropagation(); drop.classList.remove('dragover'); });
        });
        drop.addEventListener('drop', function(e){
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                renderFilePreview(e.dataTransfer.files[0]);
            }
        });
        drop.addEventListener('click', function(){ fileInput.click(); });
        fileInput.addEventListener('change', function(e){
            if(fileInput.files && fileInput.files.length){ renderFilePreview(fileInput.files[0]); }
            else clearFilePreview();
        });
    }

    if(removeFileBtn){
        removeFileBtn.addEventListener('click', function(){
            if(fileInput){ fileInput.value = ''; }
            clearFilePreview();
        });
    }

    // Link preview
    const urlInput = document.getElementById('videoUrl');
    const linkPreview = document.getElementById('linkPreview');
    function setLinkPreview(html){ if(linkPreview){ linkPreview.innerHTML = ''; linkPreview.appendChild(html); linkPreview.setAttribute('aria-hidden','false'); } }
    function clearLinkPreview(){ if(linkPreview){ linkPreview.innerHTML = ''; linkPreview.setAttribute('aria-hidden','true'); } }

    function makeEmbedFromUrl(url){
        if(!url) return null;
        // youtube
        let m = url.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|v\/))([A-Za-z0-9_-]{6,})/);
    if(m){ const iframe = document.createElement('iframe'); iframe.setAttribute('frameborder','0'); iframe.setAttribute('allow','accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture'); iframe.setAttribute('allowfullscreen',''); iframe.className = 'responsive-embed'; iframe.src = 'https://www.youtube.com/embed/' + m[1]; return iframe; }
        // vimeo
        m = url.match(/vimeo\.com\/(\d+)/);
    if(m){ const iframe = document.createElement('iframe'); iframe.setAttribute('frameborder','0'); iframe.setAttribute('allowfullscreen',''); iframe.className = 'responsive-embed'; iframe.src = 'https://player.vimeo.com/video/' + m[1]; return iframe; }
        // direct video
        try{
            const vid = document.createElement('video'); vid.controls = true; vid.preload = 'metadata'; vid.className = 'responsive-embed'; vid.src = url; return vid;
        } catch(e){ return null; }
    }

    if(urlInput){
        urlInput.addEventListener('input', function(e){
            clearLinkPreview();
            const val = (urlInput.value || '').trim();
            if(!val) return;
            const embed = makeEmbedFromUrl(val);
            if(embed){ setLinkPreview(embed); }
            else { linkPreview.textContent = 'Cannot preview this link.'; }
        });
    }
});
</script>

</body>
</html>
