# Deployment Recommendations
## For Modotte/AIRealNet FastAPI Server

Here are the deployment options for your final project (Tugas Akhir). Since you need an API to serve your Flutter app, you must host this Docker container.

### Option A: Proxmox Homelab (Your Existing Setup)
You have a VM with **4 Cores (i5-3470) and 6GB RAM**.

*   **Pros:** Free, full control, already set up.
*   **Cons:** The i5-3470 is from 2012 and lacks AVX2 instruction sets, which means ONNX Runtime will fall back to slower execution paths. Inference might take 1-2 seconds per image.
*   **Verdict:** **Perfect for development and your thesis defense demo.** As long as the mobile app shows a loading spinner, a 1-2 second delay is acceptable for a student project.
*   **How to Deploy:**
    ```bash
    # Copy project to your Proxmox VM
    # Inside the project folder, run:
    docker compose up -d
    ```

### Option B: Budget Cloud VPS (If Homelab is Too Slow)
If you find the homelab too slow, or you want the API to be accessible over the public internet without dealing with port forwarding/Cloudflare Tunnels, consider a budget VPS.

1.  **Hetzner Cloud CX22**
    *   **Specs:** 2 vCPU (modern AMD/Intel with AVX2), 4GB RAM
    *   **Cost:** ~€3.29/month (~Rp 56,000/month)
    *   **Pros:** Incredible value, very fast CPUs (AVX2 supported = fast ONNX inference).
    *   **Cons:** Datacenters are in Europe/US, so network latency to Indonesia is ~150-250ms.
2.  **Contabo VPS S**
    *   **Specs:** 4 vCPU, 6GB RAM
    *   **Cost:** ~€5.99/month (~Rp 100,000/month)
    *   **Pros:** Massive resources for the price.
3.  **AWS Free Tier / Lightsail (Not Recommended for this)**
    *   **Specs:** AWS Free Tier (t2.micro/t3.micro) only gives 1GB RAM.
    *   **Why Not?** 1GB RAM is too small for running ONNX Runtime loading a 200MB INT8 model along with the FastAPI server overhead. It will likely crash with OOM (Out of Memory) errors. AWS Lightsail 2GB costs $10/month, which is pricier than Hetzner.

### Final Recommendation for Tugas Akhir
1.  **Start with Option A (Proxmox Homelab).** Deploy the `docker-compose.yml` there. Test it with your Flutter app. If the speed is acceptable (e.g., < 2 seconds), stick with this! It costs nothing.
2.  If you need to show it working outside your home network, use **Cloudflare Tunnels (cloudflared)** on your Proxmox VM to expose `localhost:8000` to a public `https://your-api.trycloudflare.com` URL securely without opening router ports.
3.  If the inference speed on the old i5-3470 is truly unbearable, rent a **Hetzner CX22** for 1 month leading up to your defense.
