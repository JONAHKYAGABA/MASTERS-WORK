"""
MIMIC-CXR VQA Data Module

Complete integration with PhysioNet datasets:
- MIMIC-CXR-JPG: 377,110 images with structured labels (CheXpert)
- MIMIC-Ext-CXR-QBA: 42M QA pairs with scene graphs, answer bboxes

Data Flow:
==========

MIMIC-CXR-JPG (images + metadata):
    files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
    mimic-cxr-2.0.0-metadata.csv.gz  -> view_position, patient_orientation, procedure
    mimic-cxr-2.0.0-chexpert.csv.gz  -> CheXpert labels (14 categories)
    mimic-cxr-2.0.0-split.csv.gz     -> train/validate/test splits

MIMIC-Ext-CXR-QBA (QA + scene graphs):
    qa/p{XX}/p{subject_id}/s{study_id}.qa.json
        -> questions[], answers[].text, answers[].localization.bboxes
    scene_data/p{XX}/p{subject_id}/s{study_id}.scene_graph.json
        -> observations (237 entities, 310 regions, bboxes)

Usage:
======
    from data import MIMICCXRVQADataset, create_dataloader
    
    dataset = MIMICCXRVQADataset(
        mimic_cxr_path='/path/to/mimic-cxr-jpg',    # MIMIC-CXR-JPG
        mimic_qa_path='/path/to/mimic-ext-cxr-qba', # MIMIC-Ext-CXR-QBA
        split='train',
        quality_grade='A',  # Use fine-tuning grade QA pairs
    )
    
    dataloader = create_dataloader(dataset, batch_size=16)
    
    for batch in dataloader:
        # Model inputs
        images = batch['images']               # (B, 3, 224, 224) preprocessed
        input_ids = batch['input_ids']         # (B, L) question tokens
        scene_graphs = batch['scene_graphs']   # List[Dict] - processed graphs
        
        # MIMIC-CXR-JPG metadata
        view_encodings = batch['view_encodings']  # (B, 4) one-hot [PA,AP,LATERAL,OTHER]
        view_positions = batch['view_positions']  # ["PA", "AP", ...]
        patient_orientations = batch['patient_orientations']  # ["Erect", "Recumbent", ...]
        
        # Classification targets
        chexpert_labels = batch['chexpert_labels']  # (B, 14) from CheXpert CSV
        answer_idx = batch['answer_idx']            # Classification answer index
        
        # Answer generation targets (from MIMIC-Ext-CXR-QBA)
        answer_ids = batch['answer_ids']            # (B, T) tokenized answer text
        reference_answers = batch['reference_answers']  # List[str] full answer text
        
        # Visual grounding targets (from answer localization)
        gt_grounding_bboxes = batch['gt_grounding_bboxes']  # (B, 4) normalized
        gt_pointing_valid = batch['gt_pointing_valid']      # (B, 1) validity
        
        # Scene graph targets (from observations)
        gt_sg_bboxes = batch['gt_sg_bboxes']      # List[Tensor(N, 4) | None]
        gt_sg_entities = batch['gt_sg_entities']  # List[Tensor(N,) | None]
        gt_sg_regions = batch['gt_sg_regions']    # List[Tensor(N,) | None]
"""

from .mimic_cxr_dataset import (
    MIMICCXRVQADataset,
    CheXpertLabelLoader,
    SceneGraphProcessor,
    create_dataloader,
    collate_fn,
    CHEXPERT_CATEGORIES,
    QUESTION_TYPE_MAP,
)

# External datasets are optional (for cross-dataset evaluation only)
try:
    from .external_datasets import (
        VQARADDataset,
        SLAKEDataset,
    )
    EXTERNAL_DATASETS_AVAILABLE = True
except ImportError as e:
    VQARADDataset = None
    SLAKEDataset = None
    EXTERNAL_DATASETS_AVAILABLE = False

__all__ = [
    # MIMIC-CXR (Primary)
    'MIMICCXRVQADataset',
    'CheXpertLabelLoader',
    'SceneGraphProcessor',
    'create_dataloader',
    'collate_fn',
    'CHEXPERT_CATEGORIES',
    'QUESTION_TYPE_MAP',
    # External Datasets (Optional - for cross-dataset evaluation)
    'VQARADDataset',
    'SLAKEDataset',
    'EXTERNAL_DATASETS_AVAILABLE',
]
