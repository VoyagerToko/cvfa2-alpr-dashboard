# Presentation Outline

## Slide 1: Title
- Hybrid CNN + Transformer Number Plate Recognition with Cloud Deployment

## Slide 2: Problem and Objectives
- Why ALPR for parking intelligence
- Project goals and constraints

## Slide 3: End-to-End System Architecture
- Data -> CV preprocessing -> Hybrid model -> Analytics -> Deployment

## Slide 4: Dataset and Splits
- Indian number plate dataset
- 70/15/15 split
- Real-world unseen data strategy

## Slide 5: CV-based Plate Localization
- Segmentation, region growing, contours, corners
- Before/after examples

## Slide 6: Hybrid Deep Model
- CNN detector
- Transformer recognizer
- CTC decoding

## Slide 7: Training Setup and Metrics
- Losses and optimization
- Character accuracy and full plate accuracy

## Slide 8: Real-Time Inference and Tracking
- Frame pipeline
- Duplicate handling
- Zone mapping

## Slide 9: Dashboard and Insights
- Occupancy trends
- Duplicate and latency monitoring

## Slide 10: Cloud and MLOps
- Docker, Kubernetes, CI/CD, Prometheus

## Slide 11: Results and Demo
- Quantitative results
- Live/recorded demo

## Slide 12: Limitations and Future Scope
- Multi-camera fusion
- Better low-light robustness
- Continuous learning loop
