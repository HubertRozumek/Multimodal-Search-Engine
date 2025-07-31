import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Tuple
import umap

class EmbeddingVisualizer:
    """
    Class for visualization and analysis of embeddings in multimodal search engine
    """
    
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.reduced_embeddings = None
        self.clusters = None
    
    def reduce_dimensions(self, method: str = "tsne", n_components: int = 2, 
                         sample_size: int = 1000) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization
        
        Args:
            method: 'tsne', 'pca', or 'umap'
            n_components: number of target dimensions (2 or 3)
            sample_size: number of samples for visualization (for efficiency)
        """
        embeddings = self.search_engine.image_embeddings
        
        # Sampling for efficiency
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings_sample = embeddings[indices]
            products_sample = [self.search_engine.products_data[i] for i in indices]
        else:
            embeddings_sample = embeddings
            products_sample = self.search_engine.products_data
            indices = np.arange(len(embeddings))
        
        print(f"Reducing dimensions using {method.upper()} for {len(embeddings_sample)} samples...")
        
        if method.lower() == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=min(30, len(embeddings_sample)//4))
            reduced = reducer.fit_transform(embeddings_sample)
        elif method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings_sample)
        elif method.lower() == "umap":
            reducer = umap.UMAP(n_components=n_components, random_state=42,
                               n_neighbors=min(15, len(embeddings_sample)//4))
            reduced = reducer.fit_transform(embeddings_sample)
        else:
            raise ValueError("Method must be 'tsne', 'pca' or 'umap'")
        
        self.reduced_embeddings = reduced
        self.sample_indices = indices
        self.sample_products = products_sample
        
        return reduced
    
    def cluster_embeddings(self, n_clusters: int = 8) -> np.ndarray:
        """
        Perform embedding clustering
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.search_engine.image_embeddings[self.sample_indices])
        self.clusters = clusters
        
        return clusters
    
    def plot_embedding_space_2d(self, color_by: str = "category", 
                                save_path: str = None) -> go.Figure:
        """
        Create interactive 2D visualization of embedding space
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        # Prepare data
        df_viz = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'name': [p['name'] for p in self.sample_products],
            'category': [p['category'] for p in self.sample_products],
            'price': [p['price'] for p in self.sample_products],
            'color': [p['color'] for p in self.sample_products],
            'brand': [p['brand'] for p in self.sample_products],
            'description': [p['description'][:100] + '...' for p in self.sample_products]
        })
        
        # Add clusters if available
        if self.clusters is not None:
            df_viz['cluster'] = [f'Cluster {c}' for c in self.clusters]
            if color_by == "cluster":
                color_col = 'cluster'
            else:
                color_col = color_by
        else:
            color_col = color_by
        
        # Create plot
        fig = px.scatter(
            df_viz, x='x', y='y', 
            color=color_col,
            hover_data=['name', 'category', 'price', 'brand'],
            title=f'Product Embedding Space (colored by: {color_by})',
            labels={'x': 'Component 1', 'y': 'Component 2'},
            width=800, height=600
        )
        
        fig.update_traces(marker_size=8, marker_opacity=0.7)
        fig.update_layout(
            font_size=12,
            title_font_size=16,
            legend_title_font_size=14
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_similarity_heatmap(self, query_results: List[Dict], 
                               save_path: str = None) -> plt.Figure:
        """
        Create similarity heatmap for search results
        """
        if not query_results:
            return None
        
        # Prepare data
        products = [r['name'][:30] for r in query_results]  # Shortened names
        similarities = [r['similarity_score'] for r in query_results]
        categories = [r['category'] for r in query_results]
        
        # Create heatmap matrix
        similarity_matrix = np.array(similarities).reshape(-1, 1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Heatmap
        sns.heatmap(
            similarity_matrix.T, 
            xticklabels=products,
            yticklabels=['Similarity Score'],
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0, vmax=1,
            ax=ax,
            cbar_kws={'label': 'Similarity Score'}
        )
        
        # Add category colors
        category_colors = plt.cm.Set3(np.linspace(0, 1, len(set(categories))))
        category_color_map = {cat: color for cat, color in zip(set(categories), category_colors)}
        
        for i, cat in enumerate(categories):
            ax.add_patch(plt.Rectangle((i, -0.1), 1, 0.1, 
                                     facecolor=category_color_map[cat], 
                                     clip_on=False))
        
        plt.xticks(rotation=45, ha='right')
        plt.title('Product Query Similarity')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_category_distribution(self) -> Dict:
        """
        Analyze category distribution in dataset
        """
        categories = [p['category'] for p in self.search_engine.products_data]
        category_counts = pd.Series(categories).value_counts()
        
        # Price stats per category
        df = pd.DataFrame(self.search_engine.products_data)
        price_stats = df.groupby('category')['price'].agg(['mean', 'median', 'min', 'max', 'std'])
        
        return {
            'category_counts': category_counts.to_dict(),
            'price_stats': price_stats.to_dict(),
            'total_products': len(self.search_engine.products_data)
        }
    
    def plot_category_analysis(self) -> go.Figure:
        """
        Create comprehensive product category analysis
        """
        analysis = self.analyze_category_distribution()
        df = pd.DataFrame(self.search_engine.products_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Category Distribution', 'Average Prices per Category', 
                          'Price Distribution', 'Brands per Category'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Category pie chart
        categories = list(analysis['category_counts'].keys())
        counts = list(analysis['category_counts'].values())
        
        fig.add_trace(
            go.Pie(labels=categories, values=counts, name="Categories"),
            row=1, col=1
        )
        
        # 2. Average price bar chart
        avg_prices = [analysis['price_stats']['mean'][cat] for cat in categories]
        
        fig.add_trace(
            go.Bar(x=categories, y=avg_prices, name="Average Price"),
            row=1, col=2
        )
        
        # 3. Price histogram
        fig.add_trace(
            go.Histogram(x=df['price'], nbinsx=30, name="Price Distribution"),
            row=2, col=1
        )
        
        # 4. Brands per category
        brand_category = df.groupby(['category', 'brand']).size().reset_index(name='count')
        top_brands = brand_category.nlargest(10, 'count')
        
        fig.add_trace(
            go.Bar(x=top_brands['brand'], y=top_brands['count'], 
                   name="Top Brands"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Product Dataset Analysis")
        return fig
    
    def create_search_performance_report(self, search_logs: List[Dict]) -> str:
        """
        Create search performance report
        """
        if not search_logs:
            return "No search data available"
        
        df_logs = pd.DataFrame(search_logs)
        
        # Performance analysis
        avg_similarity = df_logs['avg_similarity'].mean()
        total_searches = len(df_logs)
        text_searches = len(df_logs[df_logs['search_type'] == 'text'])
        image_searches = len(df_logs[df_logs['search_type'] == 'image'])
        
        # Top queries
        if 'query' in df_logs.columns:
            top_queries = df_logs['query'].value_counts().head(5)
        else:
            top_queries = pd.Series([], dtype=object)
        
        # Category analysis in results
        all_categories = []
        for results in df_logs['results']:
            all_categories.extend([r.get('category', 'unknown') for r in results])
        
        category_performance = pd.Series(all_categories).value_counts()
        
        report = f"""
        # Search Performance Report
        
        ## General Statistics
        - **Total searches**: {total_searches}
        - **Text searches**: {text_searches} ({text_searches/total_searches*100:.1f}%)
        - **Image searches**: {image_searches} ({image_searches/total_searches*100:.1f}%)
        - **Average similarity**: {avg_similarity:.3f} ({avg_similarity*100:.1f}%)
        
        ## Top Text Queries
        """
        
        for query, count in top_queries.items():
            report += f"- {query}: {count} searches\n"
        
        report += f"""
        
        ## Performance by Category
        """
        
        for category, count in category_performance.head(5).items():
            percentage = count / len(all_categories) * 100
            report += f"- {category.title()}: {count} results ({percentage:.1f}%)\n"
        
        return report
    
    def visualize_query_similarity_distribution(self, query_results: List[Dict]) -> go.Figure:
        """
        Visualize similarity distribution for query results
        """
        if not query_results:
            return go.Figure()
        
        similarities = [r['similarity_score'] for r in query_results]
        categories = [r['category'] for r in query_results]
        prices = [r['price'] for r in query_results]
        names = [r['name'] for r in query_results]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Similarity Distribution', 'Similarity vs Price', 
                          'Similarity by Category', 'Result Ranking'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "bar"}]]
        )
        
        # 1. Similarity histogram
        fig.add_trace(
            go.Histogram(x=similarities, nbinsx=20, name="Similarity"),
            row=1, col=1
        )
        
        # 2. Scatter plot similarity vs price
        fig.add_trace(
            go.Scatter(x=similarities, y=prices, mode='markers',
                      text=names, name="Products",
                      marker=dict(size=10, opacity=0.7)),
            row=1, col=2
        )
        
        # 3. Box plot per category
        df_results = pd.DataFrame({
            'similarity': similarities,
            'category': categories,
            'price': prices
        })
        
        for category in set(categories):
            cat_similarities = df_results[df_results['category'] == category]['similarity']
            fig.add_trace(
                go.Box(y=cat_similarities, name=category),
                row=2, col=1
            )
        
        # 4. Bar chart ranking
        ranks = list(range(1, len(similarities) + 1))
        fig.add_trace(
            go.Bar(x=ranks, y=similarities, 
                   text=[name[:20] + '...' for name in names],
                   textposition='outside'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800, 
            title_text="Search Results Analysis",
            showlegend=False
        )
        
        return fig

class SearchEngineEvaluator:
    """
    Class for evaluating search engine performance
    """
    
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.evaluation_results = []
    
    def create_test_queries(self, n_queries: int = 50) -> List[Dict]:
        """
        Create test query set
        """
        products = self.search_engine.products_data
        test_queries = []
        
        # Queries based on real products
        for i in range(n_queries):
            product = np.random.choice(products)
            
            # Different query types
            query_types = [
                f"{product['color']} {product['item_type']}",
                f"{product['brand']} {product['item_type']}",
                f"{product['color']} {product['brand']}",
                product['item_type'],
                f"{product['material']} {product['item_type']}" if product['material'] else product['item_type']
            ]
            
            query = np.random.choice([q for q in query_types if q])
            
            test_queries.append({
                'query': query,
                'expected_category': product['category'],
                'expected_product_id': product['id'],
                'query_type': 'specific' if len(query.split()) > 1 else 'general'
            })
        
        return test_queries
    
    def evaluate_search_quality(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Evaluate search quality
        """
        results = {
            'precision_at_k': [],
            'category_accuracy': [],
            'avg_similarity': [],
            'query_performance': []
        }
        
        for test_query in test_queries:
            query = test_query['query']
            expected_category = test_query['expected_category']
            
            # Execute search
            search_results = self.search_engine.search_by_text(query, k=k)
            
            if search_results:
                # Precision@K for categories
                correct_category_count = sum(1 for r in search_results 
                                           if r['category'] == expected_category)
                precision = correct_category_count / len(search_results)
                results['precision_at_k'].append(precision)
                
                # Category accuracy (if top result has correct category)
                category_accuracy = 1 if search_results[0]['category'] == expected_category else 0
                results['category_accuracy'].append(category_accuracy)
                
                # Average similarity
                avg_sim = np.mean([r['similarity_score'] for r in search_results])
                results['avg_similarity'].append(avg_sim)
                
                # Per-query details
                results['query_performance'].append({
                    'query': query,
                    'expected_category': expected_category,
                    'precision': precision,
                    'category_accuracy': category_accuracy,
                    'avg_similarity': avg_sim,
                    'top_result_category': search_results[0]['category']
                })
        
        # Aggregate results
        summary = {
            'mean_precision_at_k': np.mean(results['precision_at_k']),
            'mean_category_accuracy': np.mean(results['category_accuracy']),
            'mean_avg_similarity': np.mean(results['avg_similarity']),
            'total_queries': len(test_queries),
            'detailed_results': results['query_performance']
        }
        
        return summary
    
    def generate_evaluation_report(self, evaluation_results: Dict) -> str:
        """
        Generate evaluation report
        """
        report = f"""
        # Search Engine Evaluation Report
        
        ## General Metrics
        - **Precision@5**: {evaluation_results['mean_precision_at_k']:.3f} ({evaluation_results['mean_precision_at_k']*100:.1f}%)
        - **Category Accuracy**: {evaluation_results['mean_category_accuracy']:.3f} ({evaluation_results['mean_category_accuracy']*100:.1f}%)
        - **Average Similarity**: {evaluation_results['mean_avg_similarity']:.3f} ({evaluation_results['mean_avg_similarity']*100:.1f}%)
        - **Test Count**: {evaluation_results['total_queries']}
        
        ## Results Interpretation
        
        **Precision@5** - percentage of top-5 results with correct category:
        - > 0.8: Excellent
        - 0.6-0.8: Good  
        - 0.4-0.6: Average
        - < 0.4: Needs improvement
        
        **Category Accuracy** - if top result has correct category:
        - > 0.9: Excellent
        - 0.7-0.9: Good
        - 0.5-0.7: Average
        - < 0.5: Needs improvement
        
        ## Category Analysis
        """
        
        # Per-category analysis
        category_performance = {}
        for result in evaluation_results['detailed_results']:
            cat = result['expected_category']
            if cat not in category_performance:
                category_performance[cat] = {'precision': [], 'accuracy': []}
            category_performance[cat]['precision'].append(result['precision'])
            category_performance[cat]['accuracy'].append(result['category_accuracy'])
        
        for category, metrics in category_performance.items():
            avg_precision = np.mean(metrics['precision'])
            avg_accuracy = np.mean(metrics['accuracy'])
            report += f"- **{category.title()}**: Precision={avg_precision:.3f}, Accuracy={avg_accuracy:.3f}\n"
        
        return report

# Usage example
def run_full_evaluation(search_engine):
    """
    Run full search engine evaluation with visualizations
    """
    print("Starting full evaluation...")
    
    # Initialize tools
    visualizer = EmbeddingVisualizer(search_engine)
    evaluator = SearchEngineEvaluator(search_engine)
    
    # 1. Embedding space visualization
    print("Creating embedding visualizations...")
    visualizer.reduce_dimensions(method="tsne")
    visualizer.cluster_embeddings(n_clusters=6)
    
    embedding_fig = visualizer.plot_embedding_space_2d(color_by="category")
    embedding_fig.show()
    
    # 2. Category analysis
    print("Analyzing category distribution...")
    category_fig = visualizer.plot_category_analysis()
    category_fig.show()
    
    # 3. Search quality evaluation
    print("Evaluating search quality...")
    test_queries = evaluator.create_test_queries(n_queries=100)
    evaluation_results = evaluator.evaluate_search_quality(test_queries)
    
    # 4. Evaluation report
    report = evaluator.generate_evaluation_report(evaluation_results)
    print(report)
    
    return {
        'visualizer': visualizer,
        'evaluator': evaluator,
        'evaluation_results': evaluation_results,
        'report': report
    }