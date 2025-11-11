# Video Upload Interface

This is a small PHP app that allows uploading video files or importing video links (e.g., YouTube/Vimeo or direct MP4). Uploaded files are stored in the `uploads/` folder and metadata is stored in `videos.json`.

How to run (XAMPP / Windows):

1. Place this folder into your XAMPP `htdocs` directory. By default that is `C:\xampp\htdocs\` or `D:\xampp\htdocs\`.
   - Example path: `D:\xampp\htdocs\Video Upload Interface`
2. Ensure PHP is running (start Apache in XAMPP Control Panel).
3. Open a browser and go to:

   http://localhost/Video%20Upload%20Interface/

Notes & usage:
- Uploads are limited to 200 MB by default. You can change the limit in `index.php` (the `$maxFileSize` variable).
- Allowed upload extensions: mp4, webm, ogg, mov, mkv. The server only checks the extension — for production, add stronger MIME checks.
- Video metadata and links are appended to `videos.json`.
- If you upload files, they will be saved under `uploads/` (created automatically).

Security and production notes:
- This is a simple demo. For real deployments:
  - Validate MIME types server-side.
  - Use authentication/authorization.
  - Consider storing entries in a database.
  - Serve uploaded files securely (avoid direct public exposure if they should be private).
