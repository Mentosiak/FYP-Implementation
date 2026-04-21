# Poster Pack

This folder contains implementation assets for the project poster.

## Architecture diagram template

Use this as a starting point for the software architecture diagram. You can paste it into a Mermaid renderer and then adjust box names, grouping, or arrows to match your poster style.

		flowchart TB
			subgraph EP[Entry Points]
				TS[train_supervised.py]
				SSL[train_ssl.py]
				CMP[compare_methods.py]
			end

			subgraph CFG[Configuration]
				YC[configs/*.yaml]
				LC[src/utils/config.py]
			end

			subgraph DATA[Data Layer]
				CIFAR[src/data/cifar.py]
				SUPLD[get_cifar_supervised_loaders()]
				SSLLD[get_cifar_ssl_loaders()]
				SPLIT[SplitConfig + stratified split]
				TF[Weak/strong transforms]
			end

			subgraph MOD[Model Layer]
				BUILD[src/models/builder.py]
				RES[ResNet18/34]
				WRN[WideResNet]
			end

			subgraph TRN[Training Layer]
				SUP[src/training/supervised.py]
				PL[src/training/ssl_pseudolabel.py]
				FM[src/training/ssl_fixmatch.py]
				MM[src/training/ssl_mixmatch.py]
				FX[src/training/ssl_flexmatch.py]
			end

			subgraph UTL[Utilities]
				HLP[src/utils/helpers.py]
				LOG[src/utils/logging_utils.py]
				VIS[src/utils/visualization.py]
			end

			subgraph OUT[Outputs]
				CK[checkpoints/]
				LG[logs/]
				EX[experiments/comparison/]
				PO[poster/figures/]
			end

			TS --> CFG
			SSL --> CFG
			CMP --> CFG
			CFG --> DATA
			CFG --> MOD
			DATA --> TRN
			MOD --> TRN
			TRN --> UTL
			TRN --> OUT
			UTL --> OUT

If you want a cleaner poster figure, keep the diagram to three levels only:

1. Entry points
2. Shared core pipeline
3. Outputs

That version is easier to fit on a poster, while the fuller one above is better for documentation.

## Files
- `poster_content.md`: section-by-section text draft mapped to your poster template.
- `generate_poster_figures.py`: generates poster-ready figures from benchmark CSV data.
- `poster_results_table.csv`: compact table used for plotting (generated).
- `figures/accuracy_by_budget.png`: main results figure (generated).
- `figures/improvement_over_supervised.png`: supporting gain figure (generated).

## Generate figures
Run from the repository root:

```powershell
c:/Users/momot/Documents/GitHub/FYP-Implementation/.venv/Scripts/python.exe poster/generate_poster_figures.py
```

## Data source
The script reads:
- `experiments/benchmark_summary/20260405_010605/benchmark_summary.csv`
- `experiments/comparison/*/comparison_results.json`

It keeps completed benchmark rows, merges comparison JSON reruns, and then selects
the best available value per method for the poster budgets (250, 1000, 4000 labels).
