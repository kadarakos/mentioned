--
title: Mentioned
emoji: 🦀
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Mentions
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Albert Einstein lived in the famous house.", "This book was written about him."]}' \
     https://kadarakos-mentioned.hf.space/predict
```
