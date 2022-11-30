### Construction_machinery
**Description**  
- Dataset: https://dacon.io/competitions/official/236013/data
- Goal: KD (Knowledge distillation)
- Model: BaseLine (ANN)
- Additional Loss: SoftTriple
    - Paper: < a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/
Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf">SoftTriple Loss: Deep Metric Learning Without Triplet Sampling</a>
    - Code: <a href="https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py">idstcv</a>
    - Description: <a href="https://wjddyd66.github.io/paper/Paper(11)SoftTriple/">wjddyd66 Blog</a>
- Additional KD: CRD (CONTRASTIVE REPRESENTATION DISTILLATION)
    - Paper: <a href="https://openreview.net/attachment?id=SkgpBJrtvS&name=original_pdf">CONTRASTIVE REPRESENTATION DISTILLATION</a>
    - Code: <a href="https://github.com/HobbitLong/RepDistiller">HobbitLong GitHub</a> 
    - Description: <a href="https://wjddyd66.github.io/paper/CRD(32)/">wjddyd66 Blog (1)</a>, <a href="https://wjddyd66.github.io/paper/CRD-Code(33)/">wjddyd66 Blog (2)</a>
    
**File architecture**
- Model: For define model
- Train: For train teacher & student model
- EDA.ipynb: EDA
- Base_Line, CRD, CRD_Soft, Soft_Triple.py: Train Teacher & Student
- S_Base_Line, S_CRD.py: Define Teacher with best performance -> Train only student
- Find-Best-Teacher(Studnet)-Model.ipynb: Find best model with hyperparameters
