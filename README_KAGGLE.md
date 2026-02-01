# Hallo2 on Kaggle (Friendly Guide)

This short guide is written for anyone who just wants to make Hallo2 talk on Kaggle—no deep tech background required. Follow the steps in order and you’ll end up with a finished video stored in your Kaggle workspace.

---

## 1. Prepare Your Kaggle Notebook

1. Log into [kaggle.com](https://www.kaggle.com/) and click **Code → Notebooks → New Notebook**.
2. In the right-side **Settings** panel:
   - Turn **GPU** on, pick **T4 x2** (or T4 x1 if that’s all you have).
   - Under **Internet**, choose *Enabled* (needed to pull models the first time).
   - Add the dataset **`female-voice`** (this gives you the image/audio mentioned below).
3. Press **Editor** to open the new notebook in the Classic editor.

---

## 2. Add Your Hugging Face Token Once

1. Still in the notebook, open the right-side **Settings → Secrets → Add a new secret**.
2. Give it the name **`HF_TOKEN`** (exactly) and paste your Hugging Face access token.
3. Save it—you’ll never have to paste the token into the notebook itself.

---

## 3. Pull This Repository Into Kaggle

Inside the first cell of the notebook run:

```bash
!git clone https://github.com/<your-github-username>/<your-new-repo>.git hallo2
%cd hallo2
```

> Replace the URL with the GitHub repo that contains these files.

---

## 4. Install Everything With One Script

Still inside the repo folder run:

```bash
!bash scripts/setup_kaggle.sh
```

This script:
- Pins PyTorch + CUDA to versions that work with T4 GPUs.
- Installs every other dependency without breaking NumPy, torch, or diffusers.

---

## 5. Run Inference

Execute:

```bash
!python scripts/kaggle_t4_infer.py
```

What it does automatically:
1. Reads your `HF_TOKEN` secret.
2. Downloads only the model files that fit within Kaggle’s `/kaggle/working` quota.
3. Symlinks them into `pretrained_models/`.
4. Runs the Hallo2 pipeline with:
   - Image: `/kaggle/input/female-voice/square2_512.png`
   - Audio: `/kaggle/input/female-voice/4s_ref_audio.wav`
   - 512×512 @ 25 fps, 20 DDIM steps, fp16 precision.

When the script finishes you’ll see:

```
All done! Final video saved to: /kaggle/working/hallo2_outputs/square2_512/merge_video.mp4
```

You can download the MP4 from the **Data** pane (look inside `working/hallo2_outputs/...`).

---

## 6. Swap Inputs (Optional)

- Replace the image/audio paths at the top of `scripts/kaggle_t4_infer.py`.
- Re-run steps 4 and 5 (setup can be skipped if nothing changed).

---

## 7. Common Questions

| Problem | Quick Fix |
| --- | --- |
| “No space left on device” during downloads | The script already trims duplicates, but if you still hit this, delete `/kaggle/working/hallo2_models` and re-run inference to redownload. |
| `HF_TOKEN` missing | Make sure the secret name is exactly `HF_TOKEN`. |
| Finished video missing | Check `/kaggle/working/hallo2_outputs/<image-stem>/merge_video.mp4`. |

That’s it—enjoy generating talking heads on Kaggle without wrestling with dependencies! 🎉
