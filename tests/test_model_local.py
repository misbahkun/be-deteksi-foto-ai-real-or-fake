import asyncio
from app.model import load_model, predict_image, session
import app.model as model_module
import time

async def main():
    print("Loading model...")
    model_module.session = load_model()
    
    print("Model loaded. Testing dummy data...")
    try:
        # We need a real image byte stream. 
        # Let's create a minimal valid JPEG in memory.
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (256, 256), color='red').save(buf, format='JPEG')
        img_bytes = buf.getvalue()
        
        start = time.time()
        res = await predict_image(img_bytes)
        end = time.time()
        
        print(f"Prediction: {res}")
        print(f"Time: {(end-start)*1000:.2f} ms")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
