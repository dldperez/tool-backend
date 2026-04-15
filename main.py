from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

app = FastAPI()

# ================= INVENTORY =================
inventory = [
    {"name": "Hammer", "quantity": 30},
    {"name": "Wrench", "quantity": 30},
    {"name": "Screwdriver", "quantity": 30},
    {"name": "Pliers", "quantity": 30},
    {"name": "Rope", "quantity": 30},
    {"name": "Toolbox", "quantity": 30},
    {"name": "Gasoline Can", "quantity": 30},
]

MAX_STOCK = 30

# 👷 ACTIVE BORROWS (IMPORTANT FIX)
active_borrows = []

# 📜 HISTORY
history = []

# ================= MODEL LOAD =================
def load_model():
    checkpoint = torch.load("tool_model.pth", map_location="cpu")

    classes = checkpoint["classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, classes


model, classes = load_model()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= REQUEST MODEL =================
class ItemRequest(BaseModel):
    name: str
    worker: str

# ================= INVENTORY =================
@app.get("/inventory")
def get_inventory():
    return {"inventory": inventory}

# ================= HISTORY =================
@app.get("/history")
def get_history():
    return {"history": history}

# ================= AI PREDICT =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    tool_name = classes[predicted.item()]

    return {"tool": tool_name}

# ================= BORROW =================
@app.post("/borrow")
def borrow_item(req: ItemRequest):

    for item in inventory:
        if item["name"] == req.name and item["quantity"] > 0:

            item["quantity"] -= 1

            active_borrows.append({
                "worker": req.worker,
                "tool": req.name
            })

            history.append({
                "worker": req.worker,
                "tool": req.name,
                "action": "borrow",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            return {"message": "borrowed", "item": item}

    return {"message": "not available"}

# ================= RETURN (FIXED LOGIC) =================
@app.post("/return")
def return_item(req: ItemRequest):

    # CHECK IF BORROW EXISTS
    match = None
    for b in active_borrows:
        if b["worker"] == req.worker and b["tool"] == req.name:
            match = b
            break

    if not match:
        return {"message": "invalid return - not borrowed"}

    # REMOVE FROM ACTIVE BORROWS
    active_borrows.remove(match)

    # RESTORE INVENTORY
    for item in inventory:
        if item["name"] == req.name and item["quantity"] < MAX_STOCK:
            item["quantity"] += 1

    history.append({
        "worker": req.worker,
        "tool": req.name,
        "action": "return",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return {"message": "returned", "item": req.name}