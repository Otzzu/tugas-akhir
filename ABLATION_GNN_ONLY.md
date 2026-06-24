# GNN-only Ablation Results (N series)

Split out of `ABLATION_RESULTS.md` to keep the main ablation clean. All runs here use `configs/ablation/gnn_only/` with `live_lm=none` (no LM forward, GNN-only mode). New GNN-only / N-series runs are appended HERE, not to ABLATION_RESULTS.md.

# GNN-only Ablation (N series)

`configs/ablation/gnn_only/` — `live_lm=none` (no LM forward, GNN-only mode). Base = A1 + Phase 3 L1 loss settings (drop focal γ=0, label_smoothing=0.1, cosine, wd=1e-3, patience=15). Varies graph pooling, residual, GNN block order. Purpose: isolate GNN architectural effects without LM noise.

| ID  | Run ID                                      | Config                                               | graph_pool | use_skip | block_style                                                                                 |
| --- | ------------------------------------------- | ---------------------------------------------------- | ---------- | -------- | ------------------------------------------------------------------------------------------- |
| N1  | `20260530_194019_lmgat_codebert_multiclass` | `N1_a1_l1.yaml`                                      | mean       | false    | resnet                                                                                      |
| N2  | `20260530_203000_lmgat_codebert_multiclass` | `N2_a1_l1_meanmax.yaml`                              | meanmax    | false    | resnet                                                                                      |
| N3  | `20260530_213022_lmgat_codebert_multiclass` | `N3_a1_l1_cnn.yaml`                                  | cnn        | false    | resnet                                                                                      |
| N4  | `20260530_225801_lmgat_codebert_multiclass` | `N4_a1_l1_meanmax_residual.yaml`                     | meanmax    | true     | resnet                                                                                      |
| N5  | `20260531_001117_lmgat_codebert_multiclass` | `N5_a1_l1_gnn_plus.yaml`                             | meanmax    | true     | gnn_plus                                                                                    |
| N6  | `20260531_033903_lmgat_codebert_multiclass` | `N6_a1_l1_gnn_plus_graphnorm.yaml`                   | meanmax    | true     | gnn_plus + GraphNorm                                                                        |
| N7  | `20260531_042425_lmgat_codebert_multiclass` | `N7_a1_l1_gnn_plus_elu.yaml`                         | meanmax    | true     | gnn_plus + ELU                                                                              |
| N8  | `20260531_064326_lmgat_codebert_multiclass` | `N8_a1_l1_gnn_plus_graphnorm_elu.yaml`               | meanmax    | true     | gnn_plus + GraphNorm + ELU                                                                  |
| N9  | `20260531_081742_lmgat_codebert_multiclass` | `N9_a1_l1_gnn_plus_elu_ffn.yaml`                     | meanmax    | true     | gnn_plus + ELU + FFN                                                                        |
| N10 | `20260531_110214_lmgat_codebert_multiclass` | `N10_a1_l1_gnn_plus_elu_ffn_pe.yaml`                 | meanmax    | true     | gnn_plus + ELU + FFN + RWSE-32 PE                                                           |
| N11 | `20260531_142518_lmgat_codebert_multiclass` | `N11_a1_l1_gnn_plus_elu_dim512.yaml`                 | meanmax    | true     | gnn_plus + ELU + hidden_dim=512                                                             |
| N12 | `20260531_154612_lmgat_codebert_multiclass` | `N12_a1_l1_gnn_plus_elu_dim768.yaml`                 | meanmax    | true     | gnn_plus + ELU + hidden_dim=768                                                             |
| N13 | `20260531_144339_lmgat_codebert_multiclass` | `N13_a1_l1_gnn_plus_elu_balo.yaml`                   | meanmax    | true     | gnn_plus + ELU + BalO init                                                                  |
| N14 | `20260531_212716_lmgat_codebert_multiclass` | `N14_a1_l1_gnn_plus_elu_dim512_balo.yaml`            | meanmax    | true     | N11 dim=512 + N13 BalO                                                                      |
| N15 | `20260531_204352_lmgat_codebert_multiclass` | `N15_a1_l1_gnn_plus_elu_ffn_linhead.yaml`            | meanmax    | true     | N9 FFN + linear func head (GNN+)                                                            |
| N16 | `20260601_055452_lmgat_codebert_multiclass` | `N16_a1_l1_gnn_plus_elu_ffn_linhead_balo.yaml`       | meanmax    | true     | N15 + BalO init                                                                             |
| N17 | `20260601_065912_lmgat_codebert_multiclass` | `N17_a1_l1_gnn_plus_elu_ffn_meanpool.yaml`           | mean       | true     | N15 + mean pool                                                                             |
| N18 | `20260601_055453_lmgat_codebert_multiclass` | `N18_a1_l1_gnn_plus_elu_ffn_addpool.yaml`            | add        | true     | N15 + add pool                                                                              |
| N19 | `20260601_075846_lmgat_codebert_multiclass` | `N19_a1_l1_gnn_plus_elu_ffn_maxpool.yaml`            | max        | true     | N15 + max pool                                                                              |
| N20 | `20260601_110545_lmgat_codebert_multiclass` | `N20_a1_l1_gnn_plus_elu_ffn_linhead_ginit.yaml`      | meanmax    | true     | N15 + G-Init (Kelesis 2024)                                                                 |
| N21 | `20260601_115551_lmgat_codebert_multiclass` | `N21_a1_l1_gnn_plus_elu_ffn_linhead_lsuv.yaml`       | meanmax    | true     | N15 + LSUV init (Mishkin 2016)                                                              |
| N22 | `20260601_144252_lmgat_codebert_multiclass` | `N22_a1_l1_gnn_plus_elu_ffn_linhead_L3.yaml`         | meanmax    | true     | N15 + num_layers=3 (depth-1)                                                                |
| N23 | `20260601_153557_lmgat_codebert_multiclass` | `N23_a1_l1_gnn_plus_elu_ffn_linhead_L5.yaml`         | meanmax    | true     | N15 + num_layers=5 (depth+1)                                                                |
| N24 | `20260601_165353_lmgat_codebert_multiclass` | `N24_a1_l1_gnn_plus_elu_ffn_linhead_L6.yaml`         | meanmax    | true     | N15 + num_layers=6 (depth+2)                                                                |
| N25 | `20260601_210541_lmgat_codebert_multiclass` | `N25_a1_l1_gnn_plus_elu_ffn_linhead_attnpool.yaml`   | attention  | true     | N15 + attention pool                                                                        |
| N26 | `20260601_220205_lmgat_codebert_multiclass` | `N26_a1_l1_gnn_plus_elu_ffn_linhead_crossattn.yaml`  | meanmax    | true     | N15 + cross-task attention                                                                  |
| N27 | `20260601_225416_lmgat_codebert_multiclass` | `N27_a1_l1_gnn_plus_elu_ffn_linhead_kendall.yaml`    | meanmax    | true     | N15 + Kendall uncertainty MTL                                                               |
| N28 | `20260601_232916_lmgat_codebert_multiclass` | `N28_a1_l1_gnn_plus_elu_ffn_linhead_pcgrad.yaml`     | meanmax    | true     | N15 + PCGrad (Yu 2020)                                                                      |
| N29 | `20260602_133832_lmgat_codebert_multiclass` | `N29_a1_l1_gnn_plus_elu_ffn_linhead_diagnose.yaml`   | meanmax    | true     | N15 + MTL diagnostics (num_workers=0 rerun)                                                 |
| N30 | `20260602_153635_lmgat_codebert_multiclass` | `N30_a1_l1_gnn_plus_elu_ffn_linhead_dualflow.yaml`   | dualflow   | true     | N15 + dualflow pool (num_workers=0 rerun)                                                   |
| N31 | `20260602_160417_lmgat_codebert_multiclass` | `N31_a1_l1_gnn_plus_elu_ffn_linhead_heads2.yaml`     | meanmax    | true     | N15 + heads=2 (num_workers=0 rerun)                                                         |
| N32 | `20260602_155136_lmgat_codebert_multiclass` | `N32_a1_l1_gnn_plus_elu_ffn_linhead_heads8.yaml`     | meanmax    | true     | N15 + heads=8 GATv2 default (num_workers=0 rerun)                                           |
| N33 | `20260602_170600_lmgat_codebert_multiclass` | `N33_a1_l1_gnn_plus_elu_ffn_linhead_heads16.yaml`    | meanmax    | true     | N15 + heads=16 (num_workers=0 rerun)                                                        |
| N34 | `20260603_121533_lmgat_codebert_multiclass` | `N34_a1_l1_gnn_plus_elu_ffn_linhead_norank.yaml`     | meanmax    | true     | N15 + rank_loss_weight=0 (drop rank)                                                        |
| N35 | `20260603_124325_lmgat_codebert_multiclass` | `N35_a1_l1_gnn_plus_elu_ffn_linhead_rank01.yaml`     | meanmax    | true     | N15 + rank_loss_weight=0.1 (halve rank)                                                     |
| N36 | `20260603_131456_lmgat_codebert_multiclass` | `N36_a1_l1_gnn_plus_elu_ffn_linhead_pcgrad_enc.yaml` | meanmax    | true     | N15 + PCGrad encoder-only (N28b fix)                                                        |
| N37 | `20260603_184356_lmgat_codebert_multiclass` | `N37_a1_l1_gnn_plus_elu_dim512.yaml`                 | meanmax    | true     | N15 + hidden_dim=512 (linear head)                                                          |
| N38 | `20260603_200317_lmgat_codebert_multiclass` | `N38_a1_l1_gnn_plus_elu_ffn4.yaml`                   | meanmax    | true     | N15 + ffn_expansion=4                                                                       |
| N39 | `20260603_205416_lmgat_codebert_multiclass` | `N39_a1_l1_gnn_plus_elu_moeffn.yaml`                 | meanmax    | true     | N15 + MoE-FFN (Switch, 8 experts)                                                           |
| N40 | _(not run — infeasible)_                    | `N40_a1_l1_gnn_plus_elu_gmoe.yaml`                   | meanmax    | true     | N15 + GMoE hop experts (OOM @48GB, 2-hop A@A explodes on CPG)                               |
| N41 | `20260604_143602_lmgat_codebert_multiclass` | `N41_a1_l1_gnn_plus_elu_edgemoe.yaml`                | meanmax    | true     | N15 + edge-type MoE (5 CPG relation experts)                                                |
| N42 | `20260605_042703_lmgat_codebert_multiclass` | `N42_a1_l1_rank03.yaml`                              | meanmax    | true     | N15 + rank_loss_weight=0.3                                                                  |
| N43 | `20260605_053417_lmgat_codebert_multiclass` | `N43_a1_l1_rank04.yaml`                              | meanmax    | true     | N15 + rank_loss_weight=0.4                                                                  |
| N44 | `20260605_042224_lmgat_codebert_multiclass` | `N44_a1_l1_supcon_group.yaml`                        | meanmax    | true     | N15 + SupCon group intragroup batch=64                                                      |
| N47 | `20260604_183502_lmgat_codebert_multiclass` | `N47_a1_l1_gatedgcn.yaml`                            | meanmax    | true     | N15 backbone gat to GatedGCN faithful GNN+                                                  |
| N45 | `20260606_091533_lmgat_codebert_multiclass` | `N45_a1_l1_mtl_group.yaml`                           | meanmax    | true     | N15 + MTL hierarchical group head, group loss 0.3                                           |
| N46 | `20260606_131451_lmgat_codebert_multiclass` | `N46_a1_l1_mtl_group_linear.yaml`                    | meanmax    | true     | N15 + MTL group head linear thin (fair depth vs N15)                                        |
| N48 | `20260606_163818_lmgat_codebert_multiclass` | `N48_a1_l1_jknet.yaml`                               | jknet      | true     | N15 + JK-Net pool concat all 4 layers to 1024D                                              |
| N49 | `20260606_173908_lmgat_codebert_multiclass` | `N49_a1_l1_imtl_mid2.yaml`                           | meanmax    | true     | N15 + intermediate MTL group at L2. CWE at L4                                               |
| N50 | `20260606_201553_lmgat_codebert_multiclass` | `N50_a1_l1_imtl_cwe_l3.yaml`                         | meanmax    | true     | N15 + CWE head at L3 pool. localization at L4                                               |
| N51 | `20260606_214055_lmgat_codebert_multiclass` | `N51_a1_l1_imtl_cwe_l2.yaml`                         | meanmax    | true     | N15 + CWE head at L2 pool. localization at L4                                               |
| N52 | `20260607_084632_lmgat_codebert_multiclass` | `N52_a1_l1_graph_aug.yaml`                           | jknet      | true     | N48 + structural graph aug (DropEdge 0.1, NodeDrop 0.05, FeatureMask 0.1)                   |
| N53 | `20260607_151540_lmgat_codebert_multiclass` | `N53_a1_l1_crt_n48.yaml`                             | jknet      | true     | cRT on N48 (Kang 2020). freeze backbone, reinit+retrain linear head, class-balanced sampler |
| N54 | `20260607_170046_lmgat_codebert_multiclass` | `N54_a1_l1_crt_n48_dropout.yaml`                     | jknet      | true     | N53 cRT but head dropout 0.3 not 0.0. A/B on classifier dropout                             |
| N56 | `20260608_004912_lmgat_codebert_multiclass` | `N56_a1_l1_tau_norm.yaml`                            | jknet      | true     | tau-norm on N48 (Kang 2020). post-hoc weight-norm rebalance, zero training. best tau=0      |
| N57 | `20260608_011545_lmgat_codebert_multiclass` | `N57_a1_l1_tailcalib.yaml`                           | jknet      | true     | TailCalibX on N48. synth tail feats from borrowed head cov, retrain head. 25020 synth       |
| N55 | `20260607_181127_lmgat_codebert_multiclass` | `N55_a1_l1_balanced_mixup.yaml`                      | jknet      | true     | N48 + Balanced-Mixup Remix on h_graph, alpha 0.2, full training                             |
| N58 | `20260608_032118_lmgat_codebert_multiclass` | `N58_a1_l1_crt_n48_weighted.yaml`                    | jknet      | true     | cRT on N48 keeping N48 loss class weights label smoothing. only loader differs              |
| N59 | `20260608_033323_lmgat_codebert_multiclass` | `N59_a1_l1_jknet_plain_ce.yaml`                      | jknet      | true     | N48 plain CE backbone. drop class weights label smoothing. paper stage 1                    |
| N60 | `20260608_062649_lmgat_codebert_multiclass` | `N60_a1_l1_crt_n59_plain.yaml`                       | jknet      | true     | cRT plain CE on N59 backbone. fully paper-pure two-stage                                    |
| N61 | `20260608_064712_lmgat_codebert_multiclass` | `N61_a1_l1_logit_adjust.yaml`                        | jknet      | true     | N48 + Logit Adjustment loss Menon 2021 tau 1.0                                              |
| N63 | `20260608_221234_lmgat_codebert_multiclass` | `tau_norm.py on N53 ckpt`                            | jknet      | true     | cRT+tau-norm post-hoc on N53 head. best tau 0.9. zero training                              |
| N62 | `20260608_162713_lmgat_codebert_multiclass` | `N62_a1_l1_crt_la_head.yaml`                         | jknet      | true     | cRT on N48 + Logit Adjustment head tau 0.3. balanced sampler + LA                           |
| N64 | `20260608_164827_lmgat_codebert_multiclass` | `N64_a1_l1_crt_la_head_n59.yaml`                     | jknet      | true     | cRT on N59 plain-CE backbone + LA head tau 0.3                                              |
| N65 | `20260608_170820_lmgat_codebert_multiclass` | `N65_a1_l1_flag.yaml`                                | jknet      | true     | N48 + FLAG adversarial node-feature Kong 2020 step 0.001 M 3                                |
| N66 | `20260617_072006_lmgat_codebert_multiclass` | `N66_a1_l1_jknet_max.yaml`                           | jknet      | true     | N48 + jknet_mode max element-wise max over 4 layers to 256D pool                            |
| N67 | `20260617_095428_lmgat_codebert_multiclass` | `N67_a1_l1_jknet_maxnode_sumreadout.yaml`            | jknet      | true     | N66 max node-agg + sum graph readout (jknet_readout add)                                    |

## Classification

Macro = unweighted mean across 26 classes (each CWE = equal weight).
Weighted = mean weighted by class support (frequent CWEs dominate).
For vuln detection: **macro recall** is primary — measures how well we catch each CWE type.

| ID  | Val F1    | Test F1   | Test Acc  | F1-w      | Prec      | Rec       | Prec-w    | Rec-w     | AUC-ROC   | Conf.     | Epochs |
| --- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------ |
| N1  | 0.490     | 0.451     | 0.469     | 0.466     | 0.395     | 0.459     | 0.458     | 0.461     | 0.843     | **0.583** | 63     |
| N2  | 0.486     | 0.475     | 0.490     | 0.487     | 0.500     | 0.476     | 0.511     | 0.498     | 0.856     | 0.267     | 76     |
| N3  | 0.392     | 0.420     | 0.445     | 0.447     | 0.413     | 0.462     | 0.467     | 0.455     | 0.831     | 0.489     | 99     |
| N4  | 0.523     | 0.472     | 0.501     | 0.499     | 0.477     | 0.477     | 0.512     | 0.503     | 0.884     | 0.145     | 92     |
| N5  | 0.525     | 0.486     | 0.491     | 0.498     | 0.469     | 0.515     | 0.515     | 0.475     | 0.891     | 0.421     | 63     |
| N6  | 0.531     | 0.465     | 0.509     | 0.509     | 0.433     | 0.500     | 0.535     | 0.500     | 0.903     | 0.471     | 55     |
| N7  | 0.505     | 0.450     | **0.510** | **0.511** | 0.505     | 0.473     | **0.543** | **0.527** | 0.902     | 0.482     | 55     |
| N8  | 0.492     | 0.469     | 0.493     | 0.489     | 0.444     | 0.484     | 0.526     | 0.501     | 0.893     | 0.524     | 44     |
| N9  | 0.479     | 0.407     | 0.445     | 0.454     | 0.404     | 0.405     | 0.478     | 0.437     | 0.863     | 0.368     | 72     |
| N10 | 0.465     | 0.407     | 0.465     | 0.463     | 0.419     | 0.458     | 0.493     | 0.471     | 0.872     | 0.409     | 59     |
| N11 | 0.508     | 0.480     | 0.491     | 0.494     | 0.502     | 0.481     | 0.510     | 0.500     | 0.892     | 0.514     | 36     |
| N12 | 0.494     | 0.440     | 0.500     | 0.500     | 0.427     | 0.471     | 0.511     | 0.501     | 0.899     | 0.507     | 49     |
| N13 | 0.509     | 0.471     | 0.502     | 0.497     | 0.477     | 0.447     | **0.543** | 0.495     | 0.901     | 0.532     | 44     |
| N14 | 0.521     | 0.489     | 0.493     | 0.483     | 0.460     | 0.507     | 0.522     | 0.490     | 0.901     | 0.430     | 64     |
| N15 | 0.500     | 0.523     | 0.487     | 0.483     | 0.462     | 0.510     | 0.489     | 0.482     | 0.879     | 0.337     | 78     |
| N16 | 0.514     | 0.524     | 0.481     | 0.480     | 0.521     | **0.551** | 0.489     | 0.484     | 0.889     | 0.334     | 77     |
| N17 | 0.453     | 0.416     | 0.447     | 0.447     | 0.408     | 0.470     | 0.468     | 0.452     | 0.871     | 0.425     | 80     |
| N18 | 0.328     | 0.281     | 0.328     | 0.307     | 0.272     | 0.446     | 0.346     | 0.319     | 0.852     | 0.260     | 74     |
| N19 | 0.474     | 0.484     | 0.489     | 0.488     | 0.495     | 0.484     | 0.507     | 0.489     | 0.896     | 0.292     | 61     |
| N20 | 0.506     | 0.448     | 0.485     | 0.483     | 0.551     | 0.501     | 0.498     | 0.489     | 0.896     | 0.356     | 58     |
| N21 | 0.493     | 0.445     | 0.467     | 0.458     | 0.502     | 0.482     | 0.491     | 0.473     | 0.891     | 0.336     | 60     |
| N22 | 0.493     | 0.451     | 0.491     | 0.486     | 0.515     | 0.470     | 0.499     | 0.494     | 0.891     | 0.373     | 63     |
| N23 | 0.480     | 0.439     | 0.449     | 0.446     | 0.478     | 0.483     | 0.479     | 0.469     | 0.892     | 0.344     | 67     |
| N24 | 0.469     | 0.472     | 0.473     | 0.469     | 0.517     | 0.486     | 0.477     | 0.467     | 0.883     | 0.344     | 69     |
| N25 | 0.474     | 0.421     | 0.443     | 0.443     | 0.417     | 0.447     | 0.476     | 0.469     | 0.889     | 0.428     | 67     |
| N26 | 0.475     | 0.429     | 0.487     | 0.481     | 0.510     | 0.486     | 0.491     | 0.472     | 0.893     | 0.374     | 55     |
| N27 | 0.255     | 0.249     | 0.401     | 0.376     | 0.261     | 0.241     | 0.364     | 0.377     | 0.865     | 0.217     | 42     |
| N28 | 0.483     | 0.449     | 0.475     | 0.471     | 0.467     | 0.445     | 0.486     | 0.466     | 0.886     | 0.392     | 57     |
| N29 | 0.519     | 0.535     | 0.481     | 0.480     | 0.528     | 0.525     | 0.493     | 0.479     | 0.891     | 0.323     | 86     |
| N30 | 0.465     | 0.412     | 0.446     | 0.440     | 0.453     | 0.455     | 0.461     | 0.443     | 0.872     | 0.496     | 32     |
| N31 | 0.497     | 0.443     | 0.491     | 0.487     | 0.454     | 0.402     | 0.494     | 0.472     | 0.886     | 0.344     | 75     |
| N32 | 0.470     | 0.426     | 0.442     | 0.442     | 0.409     | 0.450     | 0.478     | 0.459     | 0.889     | 0.298     | 53     |
| N33 | 0.471     | 0.433     | 0.456     | 0.456     | 0.411     | 0.496     | 0.481     | 0.471     | 0.889     | 0.314     | 55     |
| N34 | 0.470     | 0.413     | 0.445     | 0.444     | 0.462     | 0.434     | 0.479     | 0.443     | 0.889     | 0.371     | 49     |
| N35 | 0.489     | 0.457     | 0.471     | 0.460     | 0.525     | 0.469     | 0.506     | 0.470     | 0.888     | 0.405     | 53     |
| N36 | 0.469     | 0.410     | 0.425     | 0.426     | 0.350     | 0.472     | 0.441     | 0.415     | 0.889     | 0.241     | 82     |
| N37 | 0.468     | 0.441     | 0.470     | 0.470     | 0.540     | 0.509     | 0.520     | 0.475     | 0.895     | 0.322     | 55     |
| N38 | 0.488     | 0.478     | 0.482     | 0.481     | 0.532     | 0.463     | 0.518     | 0.492     | 0.891     | 0.321     | 66     |
| N39 | 0.476     | 0.461     | 0.478     | 0.472     | 0.546     | 0.525     | 0.492     | 0.475     | 0.897     | 0.383     | 55     |
| N41 | 0.465     | 0.472     | 0.485     | 0.481     | 0.498     | 0.524     | 0.504     | 0.499     | 0.894     | 0.361     | 60     |
| N42 | 0.506     | 0.468     | 0.484     | 0.480     | 0.491     | 0.466     | 0.504     | 0.500     | 0.897     | 0.344     | 77     |
| N43 | 0.502     | 0.435     | 0.459     | 0.458     | 0.469     | 0.427     | 0.489     | 0.477     | 0.889     | 0.371     | 72     |
| N44 | 0.472     | 0.422     | 0.457     | 0.456     | 0.501     | 0.389     | 0.491     | 0.430     | 0.892     | 0.399     | 49     |
| N47 | 0.475     | 0.438     | 0.474     | 0.470     | 0.440     | 0.461     | 0.488     | 0.473     | 0.898     | 0.345     | 51     |
| N45 | 0.455     | 0.435     | 0.471     | 0.471     | 0.416     | 0.438     | 0.492     | 0.473     | 0.874     | 0.368     | 76     |
| N46 | 0.462     | 0.435     | 0.470     | 0.464     | 0.484     | 0.460     | 0.493     | 0.469     | 0.873     | 0.378     | 55     |
| N48 | **0.535** | 0.525     | 0.507     | 0.507     | 0.489     | 0.509     | 0.532     | 0.524     | **0.908** | 0.462     | 40     |
| N49 | 0.483     | 0.461     | 0.473     | 0.464     | 0.475     | 0.449     | 0.490     | 0.472     | 0.889     | 0.353     | 66     |
| N50 | 0.505     | 0.449     | 0.484     | 0.481     | **0.557** | 0.493     | 0.517     | 0.483     | 0.895     | 0.360     | 65     |
| N51 | 0.494     | 0.503     | 0.484     | 0.477     | 0.517     | 0.450     | 0.499     | 0.481     | 0.896     | 0.380     | 41     |
| N52 | 0.514     | 0.494     | 0.512     | 0.511     | 0.511     | 0.496     | 0.518     | 0.512     | 0.905     | 0.488     | 44     |
| N53 | 0.511     | **0.538** | 0.528     | 0.527     | 0.529     | **0.570** | 0.533     | 0.528     | **0.912** | **0.663** | 21     |
| N54 | 0.520     | 0.479     | 0.503     | 0.499     | 0.462     | 0.529     | 0.519     | 0.503     | 0.912     | 0.517     | 13     |
| N56 | 0.532     | 0.521     | 0.504     | 0.503     | 0.546     | 0.533     | 0.520     | 0.504     | 0.908     | 0.457     | 0      |
| N57 | 0.498     | 0.523     | 0.506     | 0.506     | 0.503     | 0.573     | 0.518     | 0.506     | 0.914     | 0.468     | 0      |
| N55 | 0.523     | 0.491     | 0.473     | 0.470     | 0.499     | 0.507     | 0.489     | 0.473     | 0.901     | 0.445     | 40     |
| N58 | 0.492     | 0.493     | 0.497     | 0.496     | 0.472     | 0.570     | 0.514     | 0.497     | 0.911     | 0.412     | 12     |
| N59 | 0.533     | 0.473     | 0.510     | 0.508     | 0.509     | 0.456     | 0.512     | 0.510     | 0.916     | 0.699     | 100    |
| N60 | 0.498     | 0.466     | 0.503     | 0.499     | 0.473     | 0.488     | 0.507     | 0.503     | 0.915     | 0.700     | 21     |
| N61 | 0.480     | 0.386     | 0.297     | 0.392     | 0.467     | 0.369     | 0.676     | 0.297     | 0.908     | 0.353     | 65     |
| N63 | 0.528     | 0.486     | 0.518     | 0.516     | 0.472     | 0.522     | 0.521     | 0.518     | 0.911     | 0.348     | 0      |
| N62 | 0.472     | 0.505     | 0.511     | 0.511     | 0.456     | 0.546     | 0.522     | 0.511     | 0.910     | 0.648     | 22     |
| N64 | 0.489     | 0.461     | 0.498     | 0.497     | 0.435     | 0.479     | 0.513     | 0.498     | 0.912     | 0.683     | 20     |
| N65 | 0.529     | 0.487     | 0.503     | 0.499     | 0.504     | 0.505     | 0.516     | 0.503     | 0.902     | 0.440     | 56     |
| N66 | 0.513     | 0.483     | 0.504     | 0.498     | 0.597     | 0.582     | 0.530     | 0.519     | 0.909     | 0.369     | 58     |
| N67 | 0.386     | 0.333     | 0.285     | 0.237     | 0.464     | 0.369     | 0.479     | 0.320     | 0.816     | 0.414     | 94     |

## Statement-Level Localization

| ID  | IFA ↓     | Top-1 ↑   | Top-5 ↑   | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --- | --------- | --------- | --------- | --------- | ---------- | ------------- |
| N1  | 1.199     | 0.845     | 0.955     | 0.227     | 0.410      | 0.037         |
| N2  | 1.312     | 0.799     | 0.934     | 0.232     | 0.430      | 0.034         |
| N3  | 1.618     | 0.807     | 0.933     | 0.222     | 0.430      | 0.039         |
| N4  | 1.672     | 0.714     | 0.931     | 0.242     | 0.449      | 0.033         |
| N5  | 0.697     | 0.835     | 0.969     | 0.257     | 0.478      | 0.032         |
| N6  | 0.654     | 0.829     | 0.965     | 0.248     | 0.480      | 0.029         |
| N7  | 0.474     | 0.918     | 0.985     | 0.274     | 0.485      | 0.023         |
| N8  | 0.543     | 0.861     | 0.974     | 0.262     | 0.480      | 0.029         |
| N9  | 0.372     | 0.930     | 0.985     | 0.255     | 0.468      | 0.027         |
| N10 | 0.515     | 0.890     | 0.981     | 0.243     | 0.451      | 0.030         |
| N11 | 0.695     | 0.864     | 0.974     | 0.245     | 0.457      | 0.031         |
| N12 | 0.401     | 0.906     | 0.984     | 0.264     | 0.467      | 0.025         |
| N13 | 0.539     | 0.889     | 0.978     | 0.263     | 0.481      | 0.028         |
| N14 | 0.606     | 0.874     | 0.978     | 0.263     | **0.491**  | 0.027         |
| N15 | 0.299     | 0.940     | **0.988** | 0.271     | 0.475      | 0.024         |
| N16 | 0.385     | 0.911     | 0.981     | 0.240     | 0.451      | 0.036         |
| N17 | **0.220** | 0.928     | **0.988** | 0.257     | 0.464      | 0.029         |
| N18 | 0.485     | 0.895     | 0.978     | **0.285** | 0.488      | 0.023         |
| N19 | 0.435     | 0.895     | 0.980     | 0.238     | 0.458      | 0.034         |
| N20 | 0.351     | 0.941     | **0.988** | 0.248     | 0.440      | 0.029         |
| N21 | 0.524     | 0.883     | 0.980     | 0.214     | 0.437      | 0.043         |
| N22 | 0.423     | 0.871     | 0.980     | 0.198     | 0.413      | 0.053         |
| N23 | 0.351     | 0.936     | 0.987     | 0.261     | 0.447      | 0.027         |
| N24 | 0.369     | 0.921     | 0.982     | 0.227     | 0.423      | 0.040         |
| N25 | 0.322     | **0.950** | 0.985     | 0.242     | 0.444      | 0.031         |
| N26 | 0.370     | 0.928     | 0.984     | 0.252     | 0.439      | 0.029         |
| N27 | 0.602     | 0.865     | 0.966     | 0.253     | **0.491**  | 0.031         |
| N28 | 0.613     | 0.884     | 0.977     | 0.238     | 0.449      | 0.034         |
| N29 | 0.271     | 0.934     | 0.987     | 0.262     | 0.449      | 0.027         |
| N30 | 0.375     | 0.940     | 0.984     | 0.225     | 0.423      | 0.038         |
| N31 | 0.529     | 0.889     | 0.977     | 0.231     | 0.455      | 0.039         |
| N32 | 0.508     | 0.915     | 0.984     | 0.246     | 0.458      | 0.032         |
| N33 | 0.490     | 0.924     | 0.982     | 0.260     | 0.456      | 0.026         |
| N34 | 9.826     | 0.269     | 0.593     | 0.076     | 0.224      | 0.175         |
| N35 | 0.581     | 0.852     | 0.974     | 0.200     | 0.438      | 0.050         |
| N36 | 0.423     | 0.909     | 0.981     | 0.256     | 0.466      | 0.029         |
| N37 | 0.334     | 0.941     | 0.984     | 0.248     | 0.459      | 0.028         |
| N38 | 0.457     | 0.899     | 0.981     | 0.241     | 0.441      | 0.033         |
| N39 | 0.353     | 0.924     | 0.985     | 0.256     | 0.474      | 0.029         |
| N41 | 0.338     | 0.893     | 0.984     | 0.279     | 0.457      | **0.021**     |
| N42 | 0.343     | 0.941     | 0.985     | 0.264     | 0.460      | 0.026         |
| N43 | 0.372     | 0.939     | 0.984     | 0.257     | 0.477      | 0.028         |
| N44 | 12.423    | 0.206     | 0.554     | 0.041     | 0.213      | 0.190         |
| N47 | 0.502     | 0.922     | 0.980     | 0.240     | 0.439      | 0.034         |
| N45 | 0.498     | 0.889     | 0.980     | 0.199     | 0.419      | 0.050         |
| N46 | 0.356     | 0.934     | 0.982     | 0.257     | 0.450      | 0.027         |
| N48 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N49 | 0.429     | 0.903     | 0.982     | 0.216     | 0.431      | 0.042         |
| N50 | 0.354     | 0.886     | 0.985     | 0.219     | 0.430      | 0.042         |
| N51 | 0.442     | 0.886     | 0.975     | 0.212     | 0.426      | 0.045         |
| N52 | 0.476     | 0.933     | 0.982     | 0.266     | 0.477      | 0.026         |
| N53 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N54 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N56 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N57 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N55 | 0.324     | 0.934     | 0.985     | 0.266     | 0.467      | 0.026         |
| N58 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N59 | 0.452     | 0.900     | 0.982     | 0.249     | 0.457      | 0.032         |
| N60 | 0.452     | 0.900     | 0.982     | 0.249     | 0.457      | 0.032         |
| N61 | 0.469     | 0.895     | 0.981     | 0.247     | 0.439      | 0.031         |
| N63 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N62 | 0.310     | 0.944     | 0.985     | 0.256     | 0.439      | 0.028         |
| N64 | 0.452     | 0.900     | 0.982     | 0.249     | 0.457      | 0.032         |
| N65 | 0.435     | 0.914     | 0.985     | 0.274     | 0.479      | 0.023         |
| N66 | 0.363     | 0.927     | 0.988     | 0.234     | 0.427      | 0.037         |
| N67 | 0.419     | 0.922     | 0.984     | 0.289     | 0.499      | 0.021         |

---

## Graph-ViT / MLP-Mixer (P series)

`configs/ablation/graph_vit/` — faithful Graph-ViT/MLP-Mixer (He et al. 2023): METIS 32-patch partition → GATv2 patch-GNN + U cross-scale mix → token-mixer over patches → pool. `live_lm=none`, ml1024 node feats, offline patched `.pt` (`_gvitp32`, reference verbatim partition). P1 = attention mixer, P2 = MLP-Mixer.

| ID  | Run ID                                 | Config                           | Mixer         | n_patches | Params |
| --- | -------------------------------------- | -------------------------------- | ------------- | --------- | ------ |
| P1  | `20260610_105429_graph_vit_multiclass` | `P1_graphvit_attn.yaml`          | attention     | 32        | 4.1M   |
| P2  | `20260610_114940_graph_vit_multiclass` | `P2_graphmlpmixer.yaml`          | mlp           | 32        | 2.6M   |
| P3  | `20260610_161100_graph_vit_multiclass` | `P3_graphmlpmixer_mlp2head.yaml` | mlp (2L head) | 32        | 2.6M   |
| P4  | `20260611_094457_graph_vit_multiclass` | `P4_graphmlpmixer_p16.yaml`      | mlp           | 16        | 2.4M   |
| P5  | `20260612_051319_graph_vit_multiclass` | `P5_graphmlpmixer_p8.yaml`       | mlp           | 8         | 2.4M   |

### Classification

| ID             | Val F1 | Test F1 | Test Acc | F1-w  | AUC-ROC | Conf. | Epochs |
| -------------- | ------ | ------- | -------- | ----- | ------- | ----- | ------ |
| P1 attention   | 0.389  | 0.340   | 0.394    | 0.402 | 0.848   | 0.367 | 90     |
| P2 mlp         | 0.389  | 0.343   | 0.409    | 0.408 | 0.867   | 0.433 | 55     |
| P3 mlp 2L head | 0.377  | 0.334   | 0.399    | 0.401 | 0.801   | 0.408 | 72     |
| P4 mlp p16     | 0.424  | 0.357   | 0.418    | 0.406 | 0.849   | 0.517 | 61     |
| P5 mlp p8      | 0.431  | **0.386**| 0.450    | 0.452 | 0.831   | 0.436 | 100    |

### Statement-Level Localization

| ID             | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| -------------- | ----- | ------- | ------- | --------- | ---------- | ------------- |
| P1 attention   | 0.436 | 0.928   | 0.988   | 0.113     | 0.323      | 0.102         |
| P2 mlp         | 0.302 | 0.956   | 0.984   | 0.154     | 0.313      | 0.082         |
| P3 mlp 2L head | 0.310 | 0.958   | 0.988   | 0.144     | 0.289      | 0.101         |
| P4 mlp p16     | 0.498 | 0.943   | 0.987   | 0.129     | 0.278      | 0.108         |
| P5 mlp p8      | 0.262 | 0.955   | 0.988   | 0.094     | 0.265      | 0.151         |

Graph-ViT/MLP-Mixer **collapses classification** — Test F1 ~0.34 vs the N-series flat-GNN best ~0.52 (N48 0.525, ≈ −0.18). Patch-isolation breaks the direct message passing CPGs (small-world, diameter 5–7) rely on, so patchify hurts short-range graphs. P2 (MLP-Mixer) ≥ P1 (attention) on every metric at lower cost (2.6M vs 4.1M, 0.56h vs 0.91h) — attention over patches adds nothing. Localization stays strong (IFA 0.30–0.44, Top-1 0.93–0.96) but the pooled graph rep is too weak to classify. **Negative result: Graph-ViT not justified for CPGs (confirms diameter analysis).**

**P3 (P2 + 2-layer FuncHead readout)** — same params (2.6M, the extra readout layer is offset elsewhere), F1 0.334 vs P2 0.343 (−0.009), AUC 0.801 vs P2 0.867 (−0.066) — readout capacity does NOT recover the collapse, slightly worse if anything. Confirms the bottleneck is the pooled graph representation itself (patch-isolation), not head capacity. Localization comparable to P1/P2 (IFA 0.310, Top-1 0.958). **Closes the readout-capacity question — P-series result stands as negative.**

**P4 (P2 MLP-Mixer with n_patches 16 instead of 32)** — Test F1 0.357 vs P2 0.343 (+0.014), AUC 0.849 vs 0.867 (−0.018), at lower cost (2.4M, 0.97h, 13.1 GB). Halving the patch count (larger patches, less patch-isolation) nudges classification up marginally but stays deeply collapsed (0.357 vs N48 0.525, −0.17). Confirms the patch-count knob does not rescue Graph-ViT on CPGs — the patchify bottleneck is structural, not a granularity tuning issue. **P-series closed: Graph-ViT/MLP-Mixer not viable for CPGs across mixer type, readout depth, and patch count.**

**P5 (P2 MLP-Mixer with n_patches 8)** — Test F1 0.386 vs P4 0.357 (+0.029) vs P2 0.343, Acc 0.450 (best of P-series), F1-w 0.452, at 2.4M / 2.29h. Fewer patches keeps helping: 32→16→8 gives a **monotone** F1 0.343→0.357→0.386. Direction is consistent — larger patches = less patch-isolation = closer to plain message passing — but the limit of the trend is just N48 itself (n_patches→1 = no partition = the flat GNN). Still −0.14 vs N48 0.525 at p8, and the only knob left (shrink patches further) converges to N48 by construction, not past it. AUC slips (0.831, lowest P-series) and localization R@5%/R@20% are weakest (0.094/0.265) despite best IFA/Top-1 (0.262/0.955). **Confirms the negative result with a clean monotone: patchify only helps insofar as it undoes itself; Graph-ViT has no headroom over the flat GNN on CPGs.**

---

## Node-Masked JEPA (Q series)

`configs/ablation/jepa/` — self-supervised pretraining of the N48 GNN encoder via node-masked latent prediction (I-JEPA / GraphMAE flavor, **NO METIS** — the P-series showed patchify collapses CPGs). Online encoder sees the graph with a random 50% of node features replaced by a learned `[MASK]` token; an EMA target encoder sees the full graph; a 2-layer MLP predictor maps online→target latents at masked nodes (SmoothL1, masked nodes only). EMA weights + stop-grad prevent collapse (pure SSL, no labels). Target BN stats recalibrated on full graphs before save. Then evaluate two ways off the same encoder: **Q1 finetune** (init the N48 classifier from it, train all params) and **Q2 frozen probe** (freeze encoder + train only the linear head — the canonical JEPA SSL-quality measure). Baseline to beat: N48 (Test F1 0.525). `live_lm=none`, ml1024 node feats, same megavul dataset as the N-series. New SSL code: `src/gnn_vuln/pretrain_jepa.py` (pretrain entry) + `train.gnn_init_checkpoint`/`freeze_gnn` (downstream, reuses the cRT freeze/eval mechanism).

| ID  | Run ID     | Config                         | Mode         | Encoder init | Encoder   |
| --- | ---------- | ------------------------------ | ------------ | ------------ | --------- |
| Q0  | (pretrain) | `Q0_jepa_pretrain_n48.yaml`    | SSL pretrain | random       | trained   |
| Q1  | `20260611_140015_lmgat_codebert_multiclass` | `Q1_jepa_finetune_n48.yaml`    | finetune     | JEPA-EMA     | trainable |
| Q2  | `20260611_144416_lmgat_codebert_multiclass` | `Q2_jepa_frozenprobe_n48.yaml` | frozen probe | JEPA-EMA     | frozen    |

### Classification

| ID              | Val F1 | Test F1 | Test Acc | F1-w  | Prec  | Rec   | Prec-w | Rec-w | AUC-ROC | Conf. | Epochs |
| --------------- | ------ | ------- | -------- | ----- | ----- | ----- | ------ | ----- | ------- | ----- | ------ |
| Q1 finetune     | 0.497  | **0.475**| 0.473    | 0.471 | 0.419 | 0.470 | 0.494  | 0.480 | 0.892   | 0.424 | 51     |
| Q2 frozen probe | 0.243  | 0.272   | 0.336    | 0.327 | 0.279 | 0.260 | 0.341  | 0.306 | 0.865   | 0.253 | 34     |

### Statement-Level Localization

| ID              | IFA ↓  | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --------------- | ------ | ------- | ------- | --------- | ---------- | ------------- |
| Q1 finetune     | 0.514  | 0.867   | 0.981   | 0.226     | 0.424      | 0.039         |
| Q2 frozen probe | 23.818 | 0.171   | 0.401   | 0.053     | 0.210      | 0.189         |

---

## Tail-Strip (top18, 19-class)

`configs/ablation/gnn_only/N48_top18_jknet.yaml` — N48 (jknet + gnn_plus, GNN-only) on the **top18** dataset = benign + top-18 CWE by count (drops the ~7 dead tail CWEs <60 samples), `num_classes 19`, `ds_name_suffix _top18`, ml1024. Built CPU-only via `scripts/build_top_cwe_subset.py` (reuses base graphs, no Joern/re-embed). Tests the tail-strip lever from [[project_continual_learning_track]] Track A. **NOT directly comparable to the 26-class N-series** — fewer, better-populated classes → macro averages over 19 not 26.

| ID         | Run ID                                      | Config                    | classes | dataset    |
| ---------- | ------------------------------------------- | ------------------------- | ------- | ---------- |
| N48-top18  | `20260613_094709_lmgat_codebert_multiclass` | `N48_top18_jknet.yaml`    | 19      | ml1024_top18 |

### Classification

| ID        | Val F1 | Test F1 | Test Acc | F1-w  | Prec  | Rec   | Prec-w | Rec-w | AUC-ROC | Conf. | Epochs |
| --------- | ------ | ------- | -------- | ----- | ----- | ----- | ------ | ----- | ------- | ----- | ------ |
| N48-top18 | 0.549  | **0.553**| 0.517    | 0.515 | 0.485 | 0.577 | 0.518  | 0.508 | 0.916   | 0.474 | 69     |

### Statement-Level Localization

| ID        | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| --------- | ----- | ------- | ------- | --------- | ---------- | ------------- |
| N48-top18 | 0.543 | 0.883   | 0.978   | 0.240     | 0.450      | 0.033         |

**N48-top18 (19-class tail-strip)** — Test F1 **0.553** vs N48 26-class **0.525** (+0.028), AUC 0.916 vs 0.891. Dropping the 7 dead tail CWEs (each F1≈0 on 1-9 test samples) lifts the macro mean as expected — a **mechanical** gain from removing F1≈0 classes from the average, not a model improvement. Confirms the long-tail ceiling is data-capped: the same backbone scores higher purely by classifying fewer, learnable classes. Localization unchanged (vuln funcs identical). Use as the **scope-restricted headline** (frame as "the N most-frequent dangerous CWEs", standard practice) alongside the 26-class with-caveat result. 4.7M params, 56s/ep, 1.08 hr on RTX 5070 Ti.

---

## Node-LM Swap (unixcoder-base-nine)

`configs/ablation/gnn_only/N48_nine.yaml` — N48 (jknet + gnn_plus, GNN-only, `live_lm=none`) with node + func LM swapped from `unixcoder-base` (6 langs, no C/C++) to `unixcoder-base-nine` (9 langs incl. C/C++, matches LOSVER). ml1024, same megavul 26-class dataset/split otherwise. Sanity test: does a C-aware node-embedding LM help vs N48 base 0.525.

| ID       | Run ID                                      | Config         | classes | dataset            |
| -------- | -------------------------------------------- | -------------- | ------- | ------------------ |
| N48-nine | `20260613_203643_lmgat_codebert_multiclass` | `N48_nine.yaml` | 26      | ml1024_nine |

### Classification

| ID       | Val F1 | Test F1 | Test Acc | F1-w  | Prec  | Rec   | Prec-w | Rec-w | AUC-ROC | Conf. | Epochs |
| -------- | ------ | ------- | -------- | ----- | ----- | ----- | ------ | ----- | ------- | ----- | ------ |
| N48-nine | 0.515  | **0.484**| 0.519    | 0.517 | 0.486 | 0.483 | 0.528  | 0.520 | 0.909   | 0.469 | 50     |

### Statement-Level Localization

| ID       | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| -------- | ----- | ------- | ------- | --------- | ---------- | ------------- |
| N48-nine | 0.490 | 0.941   | 0.987   | 0.260     | 0.460      | 0.028         |

**N48-nine (unixcoder-base-nine node/func LM)** — Test F1 **0.484** vs N48 base **0.525** (−0.041), AUC 0.909 vs 0.891 (+0.018). The C-aware LM does **not** help classification — flat-to-worse macro F1, despite higher AUC and unchanged-to-slightly-better localization (IFA 0.490 vs 0.310, Top-1/Top-5 ~same, R@5%/R@20% +0.004/+0.021). Per the config's own decision rule ("if flat, stay on base") — **do not rebuild H10/O1 on nine**; stay on `unixcoder-base` for the headline LM. 4.7M params, 41s/ep, 0.57 hr on RTX 3090 Ti.

## Vuln-Only (25-class, no benign)

`configs/ablation/vulnonly/N48_vulnonly.yaml` — N48 (jknet + gnn_plus, GNN-only) on the **vuln-only** dataset = top-25 dangerous CWE, benign dropped, `num_classes 25`, `ds_name_suffix _vulnonly`, ml1024, `num_workers 0`. Built for the **apples-to-apples head-to-head vs the vuln-only baselines** (LOSVER, VulExplainer, LIVABLE) which all train without benign — see [[project_paper_baselines]] + BASELINE_RESULTS.md. Same megavul split/flaw GT as the 26-class N-series, only the label space differs.

| ID         | Run ID                                      | Config              | classes | dataset            |
| ---------- | ------------------------------------------- | ------------------- | ------- | ------------------ |
| N48-vo     | `20260617_230612_lmgat_codebert_multiclass` | `N48_vulnonly.yaml` | 25      | ml1024_vulnonly    |
| N48-vo-jk  | `20260618_152504_lmgat_codebert_multiclass` | `N48_vulnonly_jkloc.yaml` | 25      | ml1024_vulnonly    |

### Classification

| ID      | Val F1 | Test F1 | Test Acc | F1-w  | Prec  | Rec   | Prec-w | Rec-w | AUC-ROC | Conf. | Epochs |
| ------- | ------ | ------- | -------- | ----- | ----- | ----- | ------ | ----- | ------- | ----- | ------ |
| N48-vo  | 0.556  | **0.601**| 0.564    | 0.558 | 0.612 | 0.599 | 0.595  | 0.590 | 0.929   | 0.502 | 42     |
| N48-vo-jk | 0.535  | 0.552   | 0.568    | 0.565 | 0.598 | 0.564 | 0.603  | 0.583 | 0.921   | 0.489 | 42     |

### Statement-Level Localization

| ID     | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ------ | ----- | ------- | ------- | --------- | ---------- | ------------- |
| N48-vo | 0.474 | 0.845   | 0.974   | 0.203     | 0.449      | 0.049         |
| N48-vo-jk | 0.676 | 0.874   | 0.966   | 0.238     | 0.461      | 0.035         |

**N48-vo (25-class vuln-only) — beats every vuln-only baseline on macro-F1.** Test F1 **0.601** vs LOSVER **0.580**, VulExplainer **0.576**, LIVABLE **0.047** (all vuln-only 25-class, BASELINE_RESULTS.md). The GNN-only backbone tops the fully-fine-tuned LM baselines on the same label space despite a frozen node-embedding LM and ~4.7M params (vs LOSVER UniXcoder + VulExplainer GraphCodeBERT, 100M+ each). AUC 0.929 highest in the set. The 26-class headline (N48 0.525) was depressed by the benign class + dead tail — restricting to the 25 dangerous CWE the baselines also use is the fair frame. Use as **the multiclass headline vs baselines.** 4.7M params, 44s/ep, 0.52 hr on RTX 5090.

---

## Sequential Two-Stage GNN (S1)

`configs/ablation/seqgnn/` + `configs/ablation/vulnonly/` — two N48-style GNNs in sequence: stage-1 localizes (per-node suspicion) → stage-2 classifies on suspicion-augmented nodes `[x, s_i]`. Joint training (CE + MIL + rank). LOSVER analogue but jointly trained. 9.5M params (~2× N48).

| Run   | Run ID                                      | Config                      | Classes | Dataset            |
| ----- | ------------------------------------------- | --------------------------- | ------- | ------------------ |
| S1    | `20260620_080752_lmgat_seqgnn_multiclass`   | `S1_seqgnn.yaml`            | 26      | ml1024             |
| S1-vo | `20260620_073427_lmgat_seqgnn_multiclass`   | `S1_seqgnn_vulnonly.yaml`   | 25      | ml1024_vulnonly    |

### Classification

| Run   | Val F1 | Test F1 | Test Acc | F1-w  | Prec  | Rec   | Prec-w | Rec-w | AUC-ROC | Conf. | Epochs |
| ----- | ------ | ------- | -------- | ----- | ----- | ----- | ------ | ----- | ------- | ----- | ------ |
| S1    | 0.506  | 0.501   | 0.475    | 0.472 | 0.497 | 0.485 | 0.502  | 0.498 | 0.902   | 0.436 | 35     |
| S1-vo | 0.546  | **0.600**| 0.591    | 0.591 | 0.581 | 0.630 | 0.616  | 0.605 | 0.929   | 0.495 | 45     |

### Statement-Level Localization

| Run   | IFA ↓ | Top-1 ↑ | Top-5 ↑ | R@5%LOC ↑ | R@20%LOC ↑ | Effort@20%R ↓ |
| ----- | ----- | ------- | ------- | --------- | ---------- | ------------- |
| S1    | 0.612 | 0.876   | 0.977   | 0.219     | 0.445      | 0.042         |
| S1-vo | 0.330 | 0.858   | 0.988   | 0.216     | 0.458      | 0.043         |

**S1 sequential ≈ N48 single-GNN — no gain from the extra stage.** 26-class: S1 **0.501** vs N48 **0.525** (slightly below). Vuln-only 25-class: S1-vo **0.600** vs N48-vo **0.601** (tied). The localize→classify staging + 2nd GNN (9.5M, ~2× params, ~2× VRAM) does not beat the single-GNN N48 on either label space — the suspicion feed adds cost without classification gain. Localization comparable (S1-vo IFA 0.330 best in the vuln-only set). Conclusion: a valid distinct architecture (two-stage, joint-trained, LOSVER analogue) but not a win over N48 here.

---

# Training Efficiency

| Run                      | GPU             | Params | Epoch Time | Total Time (hr) | VRAM Peak |
| ------------------------ | --------------- | ------ | ---------- | --------------- | --------- |
| N1 a1+l1 mean            | RTX A4500       | 3.5M   | 47s        | 0.82            | 11.0 GB   |
| N2 a1+l1 meanmax         | RTX A4500       | 3.5M   | 47s        | 1.00            | 9.2 GB    |
| N3 a1+l1 cnn             | RTX A4500       | 4.7M   | 53s        | 1.45            | 9.1 GB    |
| N4 a1+l1 meanmax+skip    | RTX A4500       | 3.7M   | 47s        | 1.21            | 9.6 GB    |
| N5 a1+l1 gnn_plus        | RTX A4500       | 3.7M   | 47s        | 0.83            | 11.0 GB   |
| N6 N5+GraphNorm          | RTX A4500       | 3.7M   | 49s        | 0.75            | 9.5 GB    |
| N7 N5+ELU                | RTX A4500       | 3.7M   | 47s        | 0.72            | 9.3 GB    |
| N8 N5+GraphNorm+ELU      | RTX A4500       | 3.7M   | 50s        | 0.61            | 10.7 GB   |
| N9 N7+FFN                | RTX A4500       | 4.8M   | 50s        | 1.01            | 10.3 GB   |
| N10 N9+RWSE-32 PE        | RTX A4500       | 4.9M   | 52s        | 0.85            | 12.4 GB   |
| N11 N7+dim512            | RTX A6000       | 10.7M  | 95s        | 0.95            | 31.2 GB   |
| N12 N7+dim768            | RTX A6000       | 21.0M  | 118s       | 1.61            | 29.8 GB   |
| N13 N7+BalO init         | RTX A5000       | 3.7M   | 57s        | 0.70            | 10.5 GB   |
| N14 N11+BalO             | RTX A6000       | 10.7M  | 87s        | 1.55            | 17.5 GB   |
| N15 N9+linear head       | RTX A4500       | 4.7M   | 54s        | 1.17            | 8.8 GB    |
| N16 N15+BalO             | RTX A4500       | 4.7M   | 50s        | 1.07            | 12.4 GB   |
| N17 N15+mean pool        | RTX A4500       | 4.7M   | 50s        | 1.10            | 10.2 GB   |
| N18 N15+add pool         | RTX 4060 Ti     | 4.7M   | 100s       | 2.05            | 10.8 GB   |
| N19 N15+max pool         | RTX 4060 Ti     | 4.7M   | 98s        | 1.67            | 9.8 GB    |
| N20 N15+G-Init           | RTX A4500       | 4.7M   | 51s        | 0.83            | 10.5 GB   |
| N21 N15+LSUV             | RTX A4500       | 4.7M   | 51s        | 0.84            | 10.1 GB   |
| N22 N15+L=3              | RTX A4500       | 3.9M   | 50s        | 0.88            | 9.1 GB    |
| N23 N15+L=5              | RTX A4500       | 5.5M   | 69s        | 1.29            | 11.2 GB   |
| N24 N15+L=6              | RTX A4500       | 6.4M   | 79s        | 1.52            | 12.9 GB   |
| N25 N15+attn pool        | RTX A4500       | 4.7M   | 50s        | 0.93            | 9.3 GB    |
| N26 N15+cross-attn       | RTX A4500       | 5.6M   | 56s        | 0.86            | 9.8 GB    |
| N27 N15+Kendall MTL      | RTX A4500       | 4.7M   | 49s        | 0.58            | 9.5 GB    |
| N28 N15+PCGrad           | RTX A4500       | 4.7M   | 98s        | 1.56            | 9.1 GB    |
| N29 N15+MTL diag (w0)    | RTX A4500       | 4.7M   | 58s        | 1.38            | 11.2 GB   |
| N30 N15+dualflow (w0)    | RTX A4500       | 4.7M   | 51s        | 0.45            | 10.9 GB   |
| N31 N15+heads=2 (w0)     | RTX A4500       | 3.0M   | 34s        | 0.71            | 5.3 GB    |
| N32 N15+heads=8 (w0)     | RTX A6000       | 8.2M   | 83s        | 1.23            | 22.1 GB   |
| N33 N15+heads=16 (w0)    | RTX A6000       | 15.0M  | 131s       | 2.00            | 26.3 GB   |
| N34 N15+rank=0           | RTX 5070 Ti     | 4.7M   | 33s        | 0.46            | 10.4 GB   |
| N35 N15+rank=0.1         | RTX 5070 Ti     | 4.7M   | 35s        | 0.52            | 11.5 GB   |
| N36 N15+PCGrad enc       | RTX 5070 Ti     | 4.7M   | 77s        | 1.76            | 10.3 GB   |
| N37 N15+dim512           | RTX A5000       | 14.7M  | 86s        | 1.31            | 13.7 GB   |
| N38 N15+ffn4             | RTX A5000       | 5.8M   | 46s        | 0.84            | 9.7 GB    |
| N39 N15+MoE-FFN          | RTX A5000       | 12.1M  | 63s        | 0.97            | 12.0 GB   |
| N41 N15+edge-MoE         | RTX 3090        | 18.5M  | 109s       | 1.82            | 9.9 GB    |
| N42 N15+rank0.3          | RTX A4500       | 4.7M   | 52s        | 1.11            | 12.4 GB   |
| N43 N15+rank0.4          | RTX A4500       | 4.7M   | 52s        | 1.03            | 10.3 GB   |
| N44 N15+supcon group     | RTX PRO 5000 Bk | 4.8M   | 49s        | 0.67            | 17.3 GB   |
| N47 N15 GatedGCN         | RTX 3090        | 2.6M   | 26s        | 0.37            | 4.6 GB    |
| N45 N15+MTL group        | RTX A4500       | 4.9M   | 55s        | 1.16            | 9.6 GB    |
| N46 N15+MTL linear       | RTX A4000       | 4.7M   | 87s        | 1.34            | 9.7 GB    |
| N48 N15+JK-Net pool      | RTX A5000       | 4.7M   | 89s        | 1.0             | 9.6 GB    |
| N49 N15+imtl mid2        | RTX A5000       | 4.7M   | 85s        | 1.57            | 9.5 GB    |
| N50 N15+imtl_cwe l3      | RTX A4000       | 4.7M   | 78s        | 1.41            | 9.8 GB    |
| N51 N15+imtl_cwe l2      | RTX A4000       | 4.7M   | 77s        | 0.87            | 9.7 GB    |
| N52 N48+graph aug        | RTX A4000       | 4.7M   | 74s        | 0.91            | 9.1 GB    |
| N53 N48+cRT              | RTX A4000       | 4.7M   | 57s        | 0.33            | 3.1 GB    |
| N54 N48+cRT+dropout      | RTX A4000       | 4.7M   | 53s        | 0.19            | 3.8 GB    |
| N56 N48+tau-norm         | post-hoc        | 4.7M   | 0s         | 0.00            | -         |
| N57 N48+tailcalib        | post-hoc        | 4.7M   | 0s         | 0.00            | -         |
| N55 N48+bal-mixup        | RTX A4000       | 4.7M   | 83s        | 0.92            | 9.6 GB    |
| N58 N48+cRT+weighted     | RTX A4000       | 4.7M   | 57s        | 0.19            | 3.2 GB    |
| N59 N48 plain CE         | RTX A4000       | 4.7M   | 85s        | 2.35            | 10.4 GB   |
| N60 cRT on N59           | RTX A4000       | 4.7M   | 56s        | 0.33            | 3.1 GB    |
| N61 N48+logit-adjust     | RTX A4000       | 4.7M   | 83s        | 1.50            | 9.8 GB    |
| N63 cRT+tau-norm         | post-hoc        | 4.7M   | 0s         | 0.00            | -         |
| N62 cRT+LA N48           | RTX A4000       | 4.7M   | 56s        | 0.34            | 2.9 GB    |
| N64 cRT+LA N59           | RTX A4000       | 4.7M   | 58s        | 0.32            | 3.0 GB    |
| N65 N48+FLAG             | RTX A4000       | 4.7M   | 210s       | 3.26            | 9.2 GB    |
| N66 N48 jknet-max        | RTX A4000       | 4.7M   | 84s        | 1.35            | 10.5 GB   |
| N67 N66 max-node sum-rd  | RTX 5070 Ti     | 4.7M   | 36s        | 0.95            | 9.1 GB    |
| P1 graph-vit attn        | RTX 5090        | 4.1M   | 36s        | 0.91            | 17.3 GB   |
| P2 graph-vit mlp         | RTX 5090        | 2.6M   | 36s        | 0.56            | 18.0 GB   |
| P3 graph-vit mlp 2L head | RTX 5000 Ada    | 2.6M   | 80s        | 1.60            | 19.0 GB   |
| P4 graph-vit mlp p16     | RTX 5090        | 2.4M   | 57s        | 0.97            | 13.1 GB   |
| P5 graph-vit mlp p8      | RTX A6000       | 2.4M   | 82s        | 2.29            | 10.6 GB   |
| Q1 jepa finetune         | RTX A4500       | 4.7M   | 51s        | 0.72            | 10.3 GB   |
| Q2 jepa frozen probe     | RTX A4500       | 4.7M   | 27s        | 0.26            | 3.3 GB    |
| N48-top18 19-class       | RTX 5070 Ti     | 4.7M   | 56s        | 1.08            | 10.8 GB   |
| N48-nine node-LM swap    | RTX 3090 Ti     | 4.7M   | 41s        | 0.57            | 10.6 GB   |
| N48-vo 25-class          | RTX 5090        | 4.7M   | 44s        | 0.52            | 13.2 GB   |
| N48-vo-jk 25-class       | RTX 5090        | 4.7M   | 44s        | 0.52            | 13.2 GB   |
| S1 26-class              | RTX 5090        | 9.5M   | 49s        | 0.48            | 18.5 GB   |
| S1-vo 25-class           | RTX 5090        | 9.5M   | 31s        | 0.39            | 17.8 GB   |

---

## Production Models (90/10/0 split, for the API)

The three deployed API models, trained on the production split (90% train, 10% val, 0% test — no
held-out test since the whole labelled set is used; evaluation is the research split runs above).
Checkpoints on Drive `checkpoints/<run_id>_checkpoints.zip` (2026-06-23). The two `lmgat_codebert`
runs are disambiguated by size (graph_based GNN-only ≈ 71 MB vs hybrid_graph_lm live-LM ≈ 1.19 GB).

| Model API        | Arch                        | Run ID                                       | Size    |
| ---------------- | --------------------------- | -------------------------------------------- | ------- |
| graph_based      | lmgat_codebert (GNN-only)   | `20260623_042745_lmgat_codebert_multiclass`  | 71 MB   |
| sequential       | lmgat_seqgnn                | `20260623_050527_lmgat_seqgnn_multiclass`    | 137 MB  |
| hybrid_graph_lm  | lmgat_codebert (live-LM)    | `20260623_055033_lmgat_codebert_multiclass`  | 1.19 GB |
