
# Tasks

## Open
* use weaver for training a dnn (without weights) on high level features to regress ctWRe
* use weaver for training a dnn (with weights) on high level features to regress ctWRe
* learn how Andreas injected lepton features into the dense layer of ParticleNet
* normalize the LLR for finding a probabilistic confidence interval

## Closed
* technical test of the ParticlNet 
* implement and train a dnn for ctWRe in pytorch

    
# Journal

<!---
this is a template for a week entry

<details><summary><font size="4">
    07.11 - 11.11
</font></summary>

***

*text

</details>

-->

<details><summary><font size="4">
    31.01 - 04.11 
</font></summary><p>
    
***
    
* made my train and plot scripts executable in the console
* start to replicate the training in weaver (without weights)

#### histograms for different features for different eft parameters

![eft_hists](https://orothbac.web.cern.ch/Journal/3101_0411_22/eft_dnn_plots/eft_hists.png)

#### Training results:

![loss](https://orothbac.web.cern.ch/Journal/3101_0411_22/eft_dnn_plots/loss.png)

**prediction vs truth across epochs**

![truth_vs_pred](https://orothbac.web.cern.ch/Journal/3101_0411_22/eft_dnn_plots/truth_vs_pred.png)

**LLR of the DNN compared to the LLR with the test weights**

![LLR](https://orothbac.web.cern.ch/Journal/3101_0411_22/eft_dnn_plots/LLR.png)


**histogram of the dnn output with weighted quantiled bins corresponding to ctWRE=0 (sm)**

![estimator_hist](https://orothbac.web.cern.ch/Journal/3101_0411_22/eft_dnn_plots/estimator_hist.png)
    
</p></details>

<details><summary><font size="4">
    07.11 - 11.11
</font></summary>

***

* there are nan s in some branches
    
</details>
