# Wholesale Customer Segmentation using Clustering

An unsupervised learning project that groups wholesale customers by their annual spending patterns using multiple clustering algorithms.

## Dataset
- **Wholesale Customers dataset** (UCI Machine Learning Repository)
- 440 customers with annual spending on 6 product categories: Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen
- `Channel` (Horeca / Retail) and `Region` are categorical, so they are excluded from clustering and used only to validate the results

## Workflow
1. **Data preparation:** log transformation (`log1p`) followed by standardization with `StandardScaler`
2. **K-Means (k=2):** cluster centroids and comparison with the real `Channel` labels
3. **New customer assignment:** predicted clusters for 5 unseen customers
4. **Elbow method:** SSE plot for k = 1 to 10 to choose the number of clusters
5. **Hierarchical clustering:** single-link, complete-link and group-average dendrograms
6. **PCA:** reduced the data to 2D for visualization
7. **DBSCAN:** density-based clustering with noise detection
8. **Spectral Clustering:** compared side-by-side with K-Means

## Results
| Method | Outcome |
|--------|---------|
| K-Means (k=2) | 85.9% match with actual Channel labels |
| Spectral Clustering | 88.0% match, the best agreement |
| DBSCAN | Found 3 clusters, but marked 36.4% of points as noise |
| Elbow method | Supported k = 2, though cluster separation was not very strong |

**Conclusion:** K-Means is preferred overall because it is simple and easy to interpret, while Spectral Clustering achieved the highest agreement with the true channels.

## Tech Stack
Python, pandas, NumPy, scikit-learn, SciPy, matplotlib

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn scipy
```
1. Download `Wholesale customers data.csv` from the UCI repository or Kaggle
2. Update the file path in the notebook
3. Open `customer-clustering.ipynb` in Jupyter Notebook or Kaggle and run all cells

## Future Improvements
- Try Gaussian Mixture Models
- Evaluate clusters with silhouette score
- Profile each cluster to give business recommendations
