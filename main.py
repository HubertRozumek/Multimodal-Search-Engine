#!/usr/bin/env python3
"""
Multimodal Search Engine with external dataset support
Supports Fashion-MNIST, DeepFashion, custom datasets, and more
"""

import argparse
import gradio as gr
import torch
from multimodal_search_engine import MultimodalSearchEngine
from dataset_adapters import DatasetIntegrator, FashionMNISTAdapter, CustomDatasetAdapter, DeepFashionAdapter
from visualization_utils import EmbeddingVisualizer, SearchEngineEvaluator
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

class EnhancedSearchEngineApp:
    """
    Enhanced Search Engine application with external dataset support
    """
    
    def __init__(self):
        self.engine = MultimodalSearchEngine()
        self.integrator = DatasetIntegrator(self.engine)
        self.current_dataset = None
        self.dataset_info = {}
    
    def load_dataset_handler(self, dataset_type: str, dataset_path: str = "", 
                           csv_path: str = "", subset_size: int = 2000,
                           split: str = "train"):
        """Handler for loading different dataset types"""
        
        try:
            if dataset_type == "Fashion-MNIST":
                return self._load_fashion_mnist(subset_size)
                
            elif dataset_type == "Folder with categories":
                if not dataset_path:
                    return "Error: Please provide path to folder with categories", ""
                return self._load_folder_dataset(dataset_path)
                
            elif dataset_type == "CSV + Images":
                if not csv_path or not dataset_path:
                    return "Error: Please provide paths to CSV and images folder", ""
                return self._load_csv_dataset(csv_path, dataset_path)
                
            elif dataset_type == "DeepFashion":
                if not dataset_path:
                    # Use default path if not provided
                    dataset_path = "data/DeepFashion"
                return self._load_deepfashion(dataset_path, split)
                
            else:
                return "Error: Unknown dataset type", ""
                
        except Exception as e:
            return f"Error loading dataset: {str(e)}", ""
    
    def _load_fashion_mnist(self, subset_size: int):
        """Loads Fashion-MNIST"""
        adapter = self.integrator.load_fashion_mnist(subset_size=subset_size)
        
        self.current_dataset = "Fashion-MNIST"
        self.dataset_info = self.integrator.get_dataset_info()
        
        info_text = f"""
        **Fashion-MNIST loaded successfully!**
        
        **Dataset information:**
        - Products: {self.dataset_info['total_products']:,}
        - Categories: {', '.join(self.dataset_info['categories'])}
        - Embeddings: {self.dataset_info['has_embeddings']:,}
        - Images: {self.dataset_info['has_images']:,}
        
        **You can now search for products!**
        Example queries: "sneakers", "dress", "black shoes"
        """
        
        example_products = self.engine.products_data[:5]
        examples_html = "<h4>Sample products:</h4>"
        for p in example_products:
            examples_html += f"<p><strong>{p['name']}</strong> - {p['category']} (${p['price']})</p>"
        
        return info_text, examples_html
    
    def _load_folder_dataset(self, folder_path: str):
        """Loads dataset from categorized folder"""
        if not os.path.exists(folder_path):
            return f"Error: Folder {folder_path} does not exist", ""
        
        adapter = self.integrator.load_custom_dataset(folder_path)
        
        self.current_dataset = "Custom Folder"
        self.dataset_info = self.integrator.get_dataset_info()
        
        info_text = f"""
        **Folder dataset loaded successfully!**
        
        **Source:** {folder_path}
        **Dataset information:**
        - Products: {self.dataset_info['total_products']:,}
        - Categories: {', '.join(self.dataset_info['categories'])}
        - Embeddings: {self.dataset_info['has_embeddings']:,}
        """
        
        category_stats = ""
        for cat, count in self.dataset_info['category_counts'].items():
            category_stats += f"<p>• {cat}: {count} products</p>"
        
        return info_text, f"<h4>Category distribution:</h4>{category_stats}"
    
    def _load_csv_dataset(self, csv_path: str, images_path: str):
        """Loads dataset from CSV + images folder"""
        if not os.path.exists(csv_path):
            return f"Error: CSV file {csv_path} does not exist", ""
        if not os.path.exists(images_path):
            return f"Error: Images folder {images_path} does not exist", ""
        
        adapter = self.integrator.load_custom_dataset(images_path, csv_path)
        
        self.current_dataset = "CSV Dataset"
        self.dataset_info = self.integrator.get_dataset_info()
        
        info_text = f"""
        **CSV dataset loaded successfully!**
        
        **CSV:** {csv_path}
        **Images:** {images_path}
        **Dataset information:**
        - Products: {self.dataset_info['total_products']:,}
        - Categories: {', '.join(self.dataset_info['categories'])}
        """
        
        return info_text, f"<h4>Dataset ready for search!</h4>"
    
    def _load_deepfashion(self, dataset_path: str, split: str):
        """Loads DeepFashion dataset"""
        if not os.path.exists(dataset_path):
            return f"Error: DeepFashion folder {dataset_path} does not exist", ""
        
        # Create DeepFashion adapter
        adapter = DeepFashionAdapter(deepfashion_path=dataset_path)
        products_data, images = adapter.load_dataset(split=split)
        
        # Integrate with search engine
        self.engine.products_data = products_data
        self.engine.images = images
        
        # Generate embeddings
        print("Generating CLIP embeddings...")
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(images), batch_size)):
            batch_images = images[i:i+batch_size]
            inputs = self.engine.processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.engine.device) for k, v in inputs.items()}
            image_features = self.engine.model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
            embeddings.append(image_features.cpu().numpy())
        
        self.engine.image_embeddings = np.vstack(embeddings)
        print(f"Generated {len(self.engine.image_embeddings)} embeddings")
        
        # Build FAISS index
        print("Building FAISS index...")
        self.engine.build_vector_database()
        
        self.current_dataset = "DeepFashion"
        
        # Get dataset info
        categories = [p['category'] for p in products_data]
        category_counts = pd.Series(categories).value_counts().to_dict()
        
        self.dataset_info = {
            "total_products": len(products_data),
            "categories": list(set(categories)),
            "category_counts": category_counts,
            "has_images": len(images),
            "has_embeddings": len(self.engine.image_embeddings)
        }
        
        info_text = f"""
        **DeepFashion dataset loaded successfully!**
        
        **Path:** {dataset_path}
        **Split:** {split}
        **Dataset information:**
        - Products: {len(products_data):,}
        - Categories: {len(set(categories))}
        - Embeddings: {len(self.engine.image_embeddings):,}
        """
        
        # Show sample products
        example_products = products_data[:5]
        examples_html = "<h4>Sample products:</h4>"
        for p in example_products:
            examples_html += f"<p><strong>{p['name']}</strong> - {p['category']} (${p['price']})</p>"
        
        return info_text, examples_html
    
    def search_handler(self, query: str, search_type: str, uploaded_image, num_results: int):
        """Unified search handler"""
        if not self.current_dataset:
            return "Error: Please load a dataset first", "", ""
        
        try:
            if search_type == "Text" and query.strip():
                results = self.engine.search_by_text(query, k=num_results)
                title = f"Search results for: '{query}'"
                
            elif search_type == "Image" and uploaded_image is not None:
                results = self.engine.search_by_image(uploaded_image, k=num_results)
                title = "Image search results"
                query = "uploaded image"
                
            else:
                return "Error: Please enter a text query or upload an image", "", ""
            
            if not results:
                return "No results found", "", ""
            
            # Format results
            results_html, stats = self.engine.format_results_for_gradio(results)
            analysis = self.engine.get_search_analytics(query, results)
            
            return f"<h3>{title}</h3>{results_html}", stats, analysis
            
        except Exception as e:
            return f"Search error: {str(e)}", "", ""
    
    def get_dataset_statistics(self):
        """Returns current dataset statistics"""
        if not self.current_dataset:
            return "No dataset loaded"
        
        info = self.dataset_info
        
        stats_html = f"""
        <div style="padding: 15px; background: #f0f0f0; border-radius: 8px;">
            <h3>Dataset statistics: {self.current_dataset}</h3>
            <p><strong>Products:</strong> {info['total_products']:,}</p>
            <p><strong>Categories:</strong> {len(info['categories'])}</p>
            <p><strong>Embeddings:</strong> {info['has_embeddings']:,}</p>
            
            <h4>Category distribution:</h4>
        """
        
        for category, count in info['category_counts'].items():
            percentage = (count / info['total_products']) * 100
            stats_html += f"<p>• {category.title()}: {count} ({percentage:.1f}%)</p>"
        
        stats_html += "</div>"
        return stats_html
    
    def create_gradio_interface(self):
        """Creates Gradio interface with external dataset support"""
        
        with gr.Blocks(title="Multimodal Search Engine + External Datasets", 
                      theme=gr.themes.Soft()) as demo:
            
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>Multimodal Search Engine</h1>
                <p style="font-size: 18px; color: #666;">
                    Supports Fashion-MNIST, DeepFashion, custom datasets, and more!
                </p>
            </div>
            """)
            
            with gr.Tabs():
                # Tab 1: Load datasets
                with gr.Tab("Load Datasets"):
                    gr.Markdown("## Select and load a dataset")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            dataset_type = gr.Radio(
                                choices=["Fashion-MNIST", "Folder with categories", 
                                        "CSV + Images", "DeepFashion"],
                                label="Dataset type",
                                value="Fashion-MNIST"
                            )
                            
                            # Fashion-MNIST options
                            with gr.Group(visible=True) as fashion_group:
                                gr.Markdown("### Fashion-MNIST")
                                subset_size = gr.Slider(
                                    minimum=100, maximum=10000, value=2000, step=100,
                                    label="Subset size"
                                )
                            
                            # Custom dataset options
                            with gr.Group(visible=False) as custom_group:
                                gr.Markdown("### Custom dataset")
                                dataset_path = gr.Textbox(
                                    label="Path to dataset folder",
                                    placeholder="/path/to/dataset/folder"
                                )
                                csv_path = gr.Textbox(
                                    label="Path to CSV file (optional)",
                                    placeholder="/path/to/metadata.csv"
                                )
                            
                            # DeepFashion options
                            with gr.Group(visible=False) as deepfashion_group:
                                gr.Markdown("### DeepFashion")
                                deepfashion_path = gr.Textbox(
                                    label="Path to DeepFashion folder",
                                    value="data/DeepFashion",
                                    placeholder="/path/to/DeepFashion"
                                )
                                split = gr.Radio(
                                    choices=["train", "test"],
                                    label="Dataset split",
                                    value="train"
                                )
                            
                            load_btn = gr.Button("Load Dataset", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("""
                            ### Instructions
                            
                            **Fashion-MNIST:**
                            - Automatically downloaded
                            - 10 clothing categories
                            - Pre-generated product descriptions
                            
                            **Folder with categories:**
                            ```
                            dataset/
                            ├── shoes/
                            │   ├── shoe1.jpg
                            │   └── shoe2.jpg
                            ├── clothing/
                            │   ├── shirt1.jpg
                            │   └── dress1.jpg
                            ```
                            
                            **CSV + Images:**
                            CSV should contain columns:
                            - image_filename
                            - name
                            - category  
                            - description
                            - price
                            
                            **DeepFashion:**
                            - Requires manual download
                            - Place in 'data/DeepFashion'
                            - Large-scale fashion dataset
                            """)
                    
                    dataset_status = gr.Markdown("")
                    dataset_preview = gr.HTML("")
                
                # Tab 2: Search
                with gr.Tab("Search"):
                    with gr.Row():
                        with gr.Column():
                            search_type = gr.Radio(
                                choices=["Text", "Image"],
                                label="Search type",
                                value="Text"
                            )
                            
                            text_query = gr.Textbox(
                                label="Text query",
                                placeholder="e.g., 'black sneakers', 'red dress'...",
                                visible=True
                            )
                            
                            image_query = gr.Image(
                                label="Upload image",
                                type="pil",
                                visible=False
                            )
                            
                            num_results = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Number of results"
                            )
                            
                            search_btn = gr.Button("Search", variant="primary", size="lg")
                        
                        with gr.Column():
                            dataset_stats_display = gr.HTML("")
                            update_stats_btn = gr.Button("Refresh Statistics")
                    
                    search_results = gr.HTML("")
                    
                    with gr.Row():
                        search_stats = gr.Markdown("")
                        search_analysis = gr.Markdown("")
                    
                    # Dynamic field visibility
                    def update_search_interface(search_type):
                        return (
                            gr.update(visible=(search_type == "Text")),
                            gr.update(visible=(search_type == "Image"))
                        )
                    
                    search_type.change(
                        update_search_interface,
                        inputs=[search_type],
                        outputs=[text_query, image_query]
                    )
                
                # Tab 3: Dataset Explorer
                with gr.Tab("Dataset Explorer"):
                    gr.Markdown("## Dataset Explorer")
                    
                    with gr.Row():
                        with gr.Column():
                            category_filter = gr.Dropdown(
                                label="Filter by category",
                                choices=[],
                                multiselect=True
                            )
                            min_price = gr.Slider(
                                minimum=0, maximum=1000, value=0,
                                label="Minimum price"
                            )
                            max_price = gr.Slider(
                                minimum=0, maximum=1000, value=1000,
                                label="Maximum price"
                            )
                            explore_btn = gr.Button("Filter Products")
                        
                        with gr.Column():
                            gr.Markdown("""
                            ### Dataset Explorer
                            - Filter products by category
                            - Set price range
                            - Explore dataset contents
                            """)
                    
                    dataset_explorer = gr.HTML("")
                    
                    def explore_dataset(categories, min_price, max_price):
                        if not self.current_dataset:
                            return "Please load a dataset first", []
                        
                        # Get available categories
                        all_categories = list(set(p['category'] for p in self.engine.products_data))
                        
                        # Apply filters
                        filtered = [
                            p for p in self.engine.products_data
                            if (not categories or p['category'] in categories) and
                            min_price <= p.get('price', 0) <= max_price
                        ][:12]
                        
                        # Update category dropdown
                        category_choices = gr.Dropdown.update(choices=all_categories)
                        
                        if not filtered:
                            return "No products match your filters", category_choices
                        
                        html = "<div style='display: flex; flex-wrap: wrap;'>"
                        for product in filtered:
                            html += f"""
                            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 5px; width: 200px;">
                                <p><strong>{product['name'][:20]}</strong></p>
                                <p>Category: {product['category']}</p>
                                <p>Price: ${product.get('price', 'N/A')}</p>
                            </div>
                            """
                        html += "</div>"
                        
                        return html, category_choices
                    
                    explore_btn.click(
                        explore_dataset,
                        inputs=[category_filter, min_price, max_price],
                        outputs=[dataset_explorer, category_filter]
                    )
                
                # Tab 4: Information & Help
                with gr.Tab("Information"):
                    gr.Markdown("""
                    ## How to use this application
                    
                    ### 1. Load a dataset
                    - **Fashion-MNIST**: Automatic download, ideal for testing
                    - **DeepFashion**: Large-scale fashion dataset (requires download)
                    - **Custom folder**: Structure: folder/category/images
                    - **CSV + images**: Maximum metadata control
                    
                    ### 2. Search for products
                    - **Text search**: Describe what you're looking for
                    - **Image search**: Upload a photo of a similar product
                    
                    ### 3. Analyze results
                    - Check similarity scores
                    - View category distribution
                    - Analyze prices and features
                    
                    ### 4. Explore datasets
                    - Browse products by category
                    - Filter by price range
                    - View dataset statistics
                    
                    ## DeepFashion Notes
                    - Download from: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
                    - Place in 'data/DeepFashion' folder
                    - Contains over 200,000 product images
                    - Rich attribute annotations
                    
                    ## Technical requirements
                    - Python 3.8+
                    - PyTorch with MPS/CUDA support
                    - 16GB+ RAM recommended for DeepFashion
                    - 10GB+ disk space for datasets
                    """)
            
            # Dynamic UI updates for dataset type
            def update_dataset_ui(dataset_type):
                fashion_visible = (dataset_type == "Fashion-MNIST")
                custom_visible = (dataset_type in ["Folder with categories", "CSV + Images"])
                deepfashion_visible = (dataset_type == "DeepFashion")
                
                return (
                    gr.update(visible=fashion_visible),
                    gr.update(visible=custom_visible),
                    gr.update(visible=deepfashion_visible)
                )
            
            dataset_type.change(
                update_dataset_ui,
                inputs=[dataset_type],
                outputs=[fashion_group, custom_group, deepfashion_group]
            )
            
            # Connect handlers
            load_btn.click(
                self.load_dataset_handler,
                inputs=[dataset_type, dataset_path, csv_path, subset_size, split],
                outputs=[dataset_status, dataset_preview]
            )
            
            search_btn.click(
                self.search_handler,
                inputs=[text_query, search_type, image_query, num_results],
                outputs=[search_results, search_stats, search_analysis]
            )
            
            update_stats_btn.click(
                self.get_dataset_statistics,
                outputs=[dataset_stats_display]
            )
        
        return demo

def main():
    """Main application function"""
    parser = argparse.ArgumentParser(description="Multimodal Search Engine with external dataset support")
    parser.add_argument("--dataset", choices=["fashion-mnist", "custom", "deepfashion"], 
                       help="Automatically load dataset")
    parser.add_argument("--dataset-path", type=str, help="Path to dataset folder")
    parser.add_argument("--csv-path", type=str, help="Path to metadata CSV file")
    parser.add_argument("--subset-size", type=int, default=2000, 
                       help="Subset size for Fashion-MNIST")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                       help="Dataset split for DeepFashion")
    parser.add_argument("--port", type=int, default=7860, help="Interface port")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    
    args = parser.parse_args()
    
    # Create application
    app = EnhancedSearchEngineApp()
    
    # Automatic dataset loading
    if args.dataset == "fashion-mnist":
        print("Loading Fashion-MNIST automatically...")
        app._load_fashion_mnist(args.subset_size)
        
    elif args.dataset == "custom" and args.dataset_path:
        print("Loading custom dataset automatically...")
        if args.csv_path:
            app._load_csv_dataset(args.csv_path, args.dataset_path)
        else:
            app._load_folder_dataset(args.dataset_path)
            
    elif args.dataset == "deepfashion":
        print("Loading DeepFashion automatically...")
        app._load_deepfashion(args.dataset_path or "data/DeepFashion", args.split)
    
    # Quick demo mode
    if args.demo:
        print("Running quick demo...")
        run_quick_demo(app)
        return
    
    # Launch interface
    demo = app.create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=True
    )

def run_quick_demo(app):
    """Runs quick demo in console"""
    print("QUICK DEMO - Multimodal Search Engine")
    print("=" * 50)
    
    # Load Fashion-MNIST
    print("Loading Fashion-MNIST (500 samples)...")
    app._load_fashion_mnist(500)
    
    # Sample searches
    test_queries = [
        "black sneakers",
        "elegant dress", 
        "warm coat",
        "casual bag"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = app.engine.search_by_text(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']} ({result['similarity_score']*100:.1f}%)")
    
    print("\nDemo complete! Run full application:")
    print("python main_with_datasets.py")

if __name__ == "__main__":
    main()