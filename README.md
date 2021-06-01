# Anomaly Detection for Tool Wear Monitoring Using a Disentangled-Variational-Autoencoder
Demonstration of anomaly detection on the [UC Berkeley milling data set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) using a disentangled-variational-autoencoder (beta-VAE). 

The method is described in the article "Self-supervised learning for tool wear monitoring with a disentangled-variational-autoencoder" in IJHM. Link to the [preprint is here](https://github.com/tvhahn/ml-tool-wear/raw/a0e4259ae57d47c623785d907e803134fe87d651/hahn2021self.pdf). The method is also described in my MASc thesis, *[Feature Engineering and End-to-End Deep Learning in Tool Wear Monitoring](https://qspace.library.queensu.ca/handle/1974/28150)*.

I also have detailed blog posts exploring the UC Berkeley milling data set ([here](https://www.tvhahn.com/posts/milling/)),  describing how the VAE is constructed ([here](https://www.tvhahn.com/posts/building-vae/)), and analyzing the results of the anomaly detection model ([here](https://www.tvhahn.com/posts/anomaly-results/)). The code is well explained in the blog posts (and associated colab notebooks).

Feel free to cite my research if you use it in any academic research.
```
@article{hahn2021self,
  title={Self-supervised learning for tool wear monitoring with a disentangled-variational-autoencoder},
  author={Hahn, Tim Von and Mechefske, Chris K},
  journal={International Journal of Hydromechatronics},
  volume={4},
  number={1},
  pages={69--98},
  year={2021},
  publisher={Inderscience Publishers (IEL)}
}
```


## How to Run
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tvhahn/ml-tool-wear/blob/master/milling-tool-wear-beta-vae.ipynb)

[**milling-tool-wear-beta-vae.ipynb**](https://colab.research.google.com/github/tvhahn/ml-tool-wear/blob/master/milling-tool-wear-beta-vae.ipynb) is the notebook to replicate the results and make the figures. I recommend using google colab. The notebook is optimized for it, it will run in your browser, and no package installation required!

## Summary
A disentangled-variational-autoencoder, with a temporal convolutional neural network, was used to model and trend tool wear in a self-supervised manner. Anomaly detection was used to make predictions in both the input and latent spaces. The experiment was performed on the UC Berkeley milling data set. 

The method achieved a precision-recall area-under-curve (PR-AUC) score of 0.45 across all cutting parameters in the milling data set, and a top score of 0.80 for shallow depth cuts. The study presents the first known use of a disentangled-variational-autoencoder for tool wear monitoring.

**Figure 1:** The tool wear trend for case 13, generated from the latent space.

![Latent Space KL-Divergence Trend on Case 13](images/latent_space_recon_case_13_150dpi_3.png)

**Figure 2:** The distribution of the test data samples in the latent space. The y-axis shows the true state of the data samples (either healthy, degraded, or failed) and the x-axis shows the anomaly predictions (either normal or abnormal). The dotted line represents a decision threshold. Each small grey dot represents one data sample.

<p align="center">
    <img src="images/dist_latent_lowres.png" width="600">
</p>

**Figure 3:** The precision-recall curve (left) and the ROC curve (right) for the best model on the test data (using latent space anomaly detection). The no-skill model (equivalent to a random model) is plotted as a comparison. In the context of this experiment, precision is the proportion of abnormal predictions that are truly abnormal (the tool is in a failed state). Recall is the proportion of truly abnormal samples (failed) that were identified correctly.

<p align="center">
    <img src="images/prauc_lowres.png">
</p>

**Figure 4:** The PR-AUC score in the latent space, while looking at one parameter in a pair at a time.

<p align="center">
    <img src="images/prauc_params_1_600dpi.png" width="700">
</p>


**Figure 5:** An example of one cut (out of 167) from the milling data set. Six signals are collected during each cut.

<p align="center">
    <img src="images/cut_145_300dpi.png" width="800">
</p>
