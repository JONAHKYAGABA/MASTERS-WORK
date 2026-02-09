#!/usr/bin/env python3
"""
MIMIC-CXR VQA Inference Example

This example demonstrates how to:
1. Load a trained model from Hugging Face Hub or local checkpoint
2. Process a chest X-ray image
3. Ask questions and get predictions

Usage:
    python examples/inference_example.py \
        --model_path ./checkpoints/best_model \
        --image_path /path/to/chest_xray.jpg \
        --question "Is there any abnormality visible?"
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mimic_vqa_model import MIMICCXRVQAModel
from configs import load_config_from_file, get_default_config


def load_model(model_path: str, device: torch.device):
    """
    Load model from a checkpoint directory.
    
    Expected directory structure (created by save_checkpoint / final_model):
        model_path/
            pytorch_model.bin   — checkpoint dict or raw state_dict
            config.json         — model + training config
    """
    model_path = Path(model_path)
    
    # Load config
    config_path = model_path / "config.json"
    if config_path.exists():
        config = load_config_from_file(str(config_path))
    else:
        print("Config not found, using defaults")
        config = get_default_config()
    
    # Initialize model with architecture params from config
    mc = config.model
    model = MIMICCXRVQAModel(
        visual_backbone=mc.visual_backbone,
        text_encoder=mc.text_encoder,
        visual_feature_dim=mc.visual_feature_dim,
        scene_graph_dim=mc.scene_graph_dim,
        num_regions=mc.num_regions,
        num_entities=mc.num_entities,
        hidden_size=mc.hidden_size,
        num_hidden_layers=mc.num_hidden_layers,
        num_attention_heads=mc.num_attention_heads,
        sim_layers=mc.sim_layers,
        num_binary_classes=mc.num_binary_classes,
        num_category_classes=mc.num_category_classes,
        num_region_classes=mc.num_region_classes,
        num_severity_classes=mc.num_severity_classes,
        dropout=mc.hidden_dropout_prob,
        use_chexpert_head=True,
    )
    
    # Load weights — handle our checkpoint format
    weights_path = model_path / "pytorch_model.bin"
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Our checkpoints wrap the state dict in a top-level dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', '?')
            step = checkpoint.get('global_step', '?')
            print(f"Loading checkpoint from epoch {epoch}, step {step}")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the file IS the state dict
            state_dict = checkpoint
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys (new heads?): {missing[:3]}...")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys: {unexpected[:3]}...")
        print(f"Loaded model from {weights_path}")
    else:
        print("Warning: No weights found, using randomly initialized model")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def preprocess_image(image_path: str, size: int = 224):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image.size


def get_question_type(question: str) -> str:
    """Infer question type from question text."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['abnormal', 'normal', 'finding', 'present']):
        return 'is_abnormal'
    elif any(word in question_lower for word in ['where', 'location', 'region']):
        return 'where_is_finding'
    elif any(word in question_lower for word in ['what', 'describe', 'diagnosis']):
        return 'describe_finding'
    elif any(word in question_lower for word in ['severe', 'severity', 'how bad']):
        return 'how_severe'
    else:
        return 'is_abnormal'  # Default to binary question


def predict(
    model: MIMICCXRVQAModel,
    image: torch.Tensor,
    question: str,
    image_size: tuple,
    device: torch.device
):
    """Run inference on an image with a question."""
    
    # Tokenize question
    question_inputs = model.tokenizer(
        question,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Get question type
    question_type = get_question_type(question)
    
    # Minimal scene graph placeholder.
    # At inference the model automatically generates a scene graph from
    # the image via SceneGraphGenerator and feeds it back into the
    # SceneGraphEncoder for multimodal fusion.  This placeholder only
    # provides a default bbox for the visual encoder's ROI pooling.
    import numpy as np
    scene_graphs = [{
        'bboxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        'region_ids': np.array([0], dtype=np.int64),
        'entity_ids': np.array([0], dtype=np.int64),
        'positiveness': np.array([0], dtype=np.int64),
        'num_objects': 1
    }]
    
    # Move to device
    image = image.to(device)
    input_ids = question_inputs['input_ids'].to(device)
    attention_mask = question_inputs['attention_mask'].to(device)
    token_type_ids = question_inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
    image_widths = torch.tensor([image_size[0]], dtype=torch.float).to(device)
    image_heights = torch.tensor([image_size[1]], dtype=torch.float).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            images=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            scene_graphs=scene_graphs,
            question_types=[question_type],
            image_widths=image_widths,
            image_heights=image_heights
        )
    
    # Parse outputs (outputs is a dict)
    results = {
        'question': question,
        'question_type': question_type,
    }
    
    # --- PRIMARY ANSWER: Decoder-generated free-form text ---
    generated_text = outputs.get('generated_answer_text')
    if generated_text:
        results['generated_answer'] = generated_text[0]
    
    # --- CLASSIFICATION ANSWER (backup): VQA head prediction ---
    from data.mimic_cxr_dataset import QUESTION_TYPE_MAP
    vqa_logits = outputs.get('vqa_logits', {})
    
    head_type = QUESTION_TYPE_MAP.get(question_type, 'binary')
    if head_type in vqa_logits:
        logits = vqa_logits[head_type]
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
        
        if head_type == 'binary':
            answer = 'Yes' if pred_class == 1 else 'No'
        elif head_type == 'severity':
            severity_levels = ['None', 'Mild', 'Moderate', 'Severe']
            answer = severity_levels[pred_class]
        else:
            answer = f"Class {pred_class}"
        
        results['classification_answer'] = answer
        results['classification_confidence'] = confidence
        results['probabilities'] = probs[0].cpu().numpy().tolist()
    
    # --- GENERATED SCENE GRAPH (from SceneGraphGenerator) ---
    generated_sgs = outputs.get('generated_scene_graphs')
    if generated_sgs:
        sg = generated_sgs[0]
        results['generated_scene_graph'] = {
            'num_objects': sg['num_objects'],
            'entity_ids': sg['entity_ids'].tolist(),
            'region_ids': sg['region_ids'].tolist(),
            'bboxes': sg['bboxes'].tolist(),
        }
    
    # --- VISUAL GROUNDING ---
    grounding = outputs.get('grounding_outputs')
    if grounding:
        results['grounding_bbox'] = grounding['bbox_pred'][0].cpu().numpy().tolist()
        results['pointing_score'] = grounding['pointing_score'][0].item()
    
    # --- CHEXPERT FINDINGS ---
    chexpert_logits = outputs.get('chexpert_logits')
    if chexpert_logits is not None:
        chexpert_probs = torch.sigmoid(chexpert_logits)
        chexpert_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'Pleural Effusion', 'Pneumonia',
            'Pneumothorax', 'Pleural Other', 'Support Devices', 'No Finding'
        ]
        
        findings = {}
        for i, label in enumerate(chexpert_labels):
            prob = chexpert_probs[0, i].item()
            findings[label] = round(prob, 3)
        
        results['chexpert_findings'] = findings
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MIMIC-CXR VQA Inference")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint or HF Hub ID')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to chest X-ray image')
    parser.add_argument('--question', type=str, default="Is there any abnormality?",
                       help='Question to ask about the image')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, config = load_model(args.model_path, device)
    
    # Load and preprocess image
    print(f"Processing image: {args.image_path}")
    image, image_size = preprocess_image(args.image_path)
    
    # Run inference
    print(f"Question: {args.question}")
    results = predict(model, image, args.question, image_size, device)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Question: {results['question']}")
    print(f"Question Type: {results['question_type']}")
    
    if 'generated_answer' in results:
        print(f"\n[Decoder Answer] {results['generated_answer']}")
    
    if 'classification_answer' in results:
        print(f"[Classification] {results['classification_answer']} "
              f"(confidence: {results.get('classification_confidence', 0):.1%})")
    
    if 'generated_scene_graph' in results:
        sg = results['generated_scene_graph']
        print(f"\n[Scene Graph] {sg['num_objects']} detected objects")
        for i in range(min(sg['num_objects'], 5)):
            bbox = sg['bboxes'][i]
            print(f"  Object {i}: entity={sg['entity_ids'][i]}, "
                  f"region={sg['region_ids'][i]}, "
                  f"bbox=[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")
    
    if 'grounding_bbox' in results:
        bbox = results['grounding_bbox']
        print(f"\n[Grounding] bbox=[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}], "
              f"pointing={results.get('pointing_score', 0):.2f}")
    
    if 'chexpert_findings' in results:
        print("\n[CheXpert Findings] (top-5 by probability):")
        for label, prob in sorted(results['chexpert_findings'].items(), key=lambda x: -x[1])[:5]:
            print(f"  {label}: {prob:.1%}")
    
    print("=" * 60)
    
    # Save full results to JSON
    output_path = Path(args.image_path).stem + "_prediction.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == '__main__':
    main()

