# Final Report Template

## 1. Abstract
- Project objective
- Core contributions
- Final outcomes

## 2. Problem Statement
- Business context (parking management)
- Technical problem definition

## 3. Dataset and Data Engineering
- Source dataset details
- Data split (70/15/15)
- Annotation verification approach
- Real-world unseen data collection process

## 4. Classical CV Preprocessing
- Threshold segmentation
- Region growing
- Canny + contour pipeline
- Harris corner detector use
- Combined boundary approximation and perspective correction

## 5. Hybrid Model Design
- Detector architecture (CNN)
- Recognizer architecture (CNN + Transformer)
- Sequence decoding (CTC)
- Hyperparameters

## 6. Training and Evaluation
- Training setup and compute environment
- Losses and optimization
- Metrics: Character Accuracy, Full Plate Accuracy, IoU
- Test and unseen real-world performance

## 7. Post-processing and Analytics
- Duplicate detection logic
- Zone mapping and occupancy estimation
- Tracking across frames

## 8. Deployment and MLOps
- API architecture
- Docker and Kubernetes setup
- CI/CD workflow
- Monitoring and drift strategy

## 9. Results and Discussion
- Success cases
- Failure modes
- Error analysis

## 10. Conclusion and Future Work
- OCR improvements
- Multi-plate scene handling
- Active learning loop
