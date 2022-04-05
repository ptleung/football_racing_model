# Football Racing Model
This is a case study for applying data science in football racing with uncertainty quantification using Variational AutoEncoder and LSTM Variational AutoEncoder.

### Model and business case desciption
For model description and business case report details, please go to Reports/JC business case report.docx and Reports/Model Specification.docx

### How to use the model
To reproduce the model, you may run codes/load_model_n_reproduce_results.py

Note that the reproduced results will be different from the documented as random samples are taken from the latent space of VAE and LSTMVAE. But the POC of Distribution of Confidence comparison (as in the distplot shown in 'Results/Sample_std_of_correct_vs_incorrect_prediction.png') will not vary a lot.
