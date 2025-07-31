import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import faiss
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import os
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class MultimodalSearchEngine:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize multimodal search engine
        """
        print("Initializing Multimodal Search Engine...")
        
        # Check MPS (Metal Performance Shaders) availability on macOS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Metal) for GPU acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for GPU acceleration")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (no GPU acceleration)")
        
        # Load CLIP model
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize variables
        self.image_embeddings = None
        self.text_embeddings = None
        self.products_data = []
        self.faiss_index = None
        self.embedding_dim = 512  # CLIP ViT-B/32 uses 512-dimensional embeddings
        
        print("CLIP model loaded successfully!")
    
    def create_sample_dataset(self, num_products: int = 1000):
        """
        Create sample product dataset (simulates real data)
        """
        print(f"Creating sample dataset with {num_products} products...")
        
        # Product categories
        categories = {
            "clothing": ["t-shirt", "jeans", "dress", "jacket", "sweater", "shorts", "skirt", "blouse"],
            "shoes": ["sneakers", "boots", "sandals", "heels", "flats", "loafers", "running shoes"],
            "accessories": ["watch", "bag", "hat", "sunglasses", "belt", "scarf", "necklace"],
            "electronics": ["laptop", "phone", "headphones", "tablet", "camera", "speaker"],
        }
        
        colors = ["black", "white", "blue", "red", "green", "brown", "gray", "navy", "beige", "pink"]
        materials = ["cotton", "leather", "denim", "wool", "silk", "polyester", "metal", "plastic"]
        brands = ["Nike", "Adidas", "Zara", "H&M", "Apple", "Samsung", "Sony", "Gucci", "Prada"]
        
        products = []
        
        for i in range(num_products):
            category = np.random.choice(list(categories.keys()))
            item_type = np.random.choice(categories[category])
            color = np.random.choice(colors)
            material = np.random.choice(materials) if category in ["clothing", "shoes", "accessories"] else None
            brand = np.random.choice(brands)
            price = np.random.randint(20, 500)
            
            # Generate text description
            description_parts = [color, item_type]
            if material and np.random.random() > 0.5:
                description_parts.insert(1, material)
            
            name = f"{brand} {' '.join(description_parts).title()}"
            description = f"{color.title()} {item_type} made of {material if material else 'high-quality materials'}. Perfect for casual and formal occasions. Brand: {brand}. Price: ${price}"
            
            # Add extra details
            if category == "clothing":
                sizes = ["XS", "S", "M", "L", "XL"]
                available_sizes = np.random.choice(sizes, size=np.random.randint(2, 5), replace=False)
                description += f". Available sizes: {', '.join(available_sizes)}"
            
            products.append({
                "id": i,
                "name": name,
                "category": category,
                "item_type": item_type,
                "color": color,
                "material": material,
                "brand": brand,
                "price": price,
                "description": description,
                "image_url": f"https://via.placeholder.com/300x300/{np.random.choice(['FF0000', '00FF00', '0000FF', 'FFFF00', 'FF00FF'])}/FFFFFF?text={item_type.replace(' ', '+')}"
            })
        
        self.products_data = products
        print(f"Dataset created with {len(products)} products!")
        return products
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode list of texts to CLIP embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                text_features = F.normalize(text_features, p=2, dim=1)
                embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode single image to CLIP embedding
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=1)
            return image_features.cpu().numpy()
    
    def create_dummy_image_embeddings(self):
        """
        Create dummy image embeddings (in real project you'd load real images)
        """
        print("Generating image embeddings (dummy data)...")
        
        # Generate random but realistic embeddings
        np.random.seed(42)  # For reproducible results
        num_products = len(self.products_data)
        
        # Create embeddings based on product features
        embeddings = []
        for product in self.products_data:
            # Encode text description as image proxy
            text_embedding = self.encode_texts([product['description']])[0]
            
            # Add noise for diversity
            noise = np.random.normal(0, 0.1, self.embedding_dim)
            image_embedding = text_embedding + noise
            
            # Normalize
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            embeddings.append(image_embedding)
        
        self.image_embeddings = np.array(embeddings)
        print(f"Generated {len(embeddings)} image embeddings!")
    
    def build_vector_database(self):
        """
        Build FAISS vector database
        """
        print("Building FAISS vector database...")
        
        if self.image_embeddings is None:
            raise ValueError("Image embeddings not created!")
        
        # Create FAISS index
        if faiss.get_num_gpus() > 0 and self.device.type in ["cuda", "mps"]:
            # GPU-accelerated index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            if self.device.type == "cuda":
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        else:
            # CPU index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.faiss_index.add(self.image_embeddings.astype('float32'))
        
        print(f"Vector database created with {self.faiss_index.ntotal} embeddings!")
    
    def search_by_text(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search products by text query
        """
        if self.faiss_index is None:
            raise ValueError("Vector database not created!")
        
        # Encode text query
        query_embedding = self.encode_texts([query])[0]
        
        # Find nearest neighbors
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.products_data):
                product = self.products_data[idx].copy()
                product['similarity_score'] = float(score)
                product['rank'] = i + 1
                results.append(product)
        
        return results
    
    def search_by_image(self, image: Image.Image, k: int = 5) -> List[Dict]:
        """
        Search products by image
        """
        if self.faiss_index is None:
            raise ValueError("Vector database not created!")
        
        # Encode query image
        query_embedding = self.encode_image(image)[0]
        
        # Find nearest neighbors
        scores, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.products_data):
                product = self.products_data[idx].copy()
                product['similarity_score'] = float(score)
                product['rank'] = i + 1
                results.append(product)
        
        return results
    
    def format_results_for_gradio(self, results: List[Dict]) -> Tuple[str, str]:
        """
        Format results for Gradio display
        """
        if not results:
            return "No results found", ""
        
        # Results table
        results_html = """
        <div style="max-width: 1000px;">
            <h3>Search Results:</h3>
        """
        
        for result in results:
            similarity_percent = result['similarity_score'] * 100
            color = "green" if similarity_percent > 80 else "orange" if similarity_percent > 60 else "red"
            
            results_html += f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background: #f9f9f9;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #333;">{result['name']}</h4>
                    <span style="background: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">
                        #{result['rank']} | {similarity_percent:.1f}% similarity
                    </span>
                </div>
                <p style="margin: 5px 0; color: #666;"><strong>Category:</strong> {result['category'].title()} | <strong>Price:</strong> ${result['price']}</p>
                <p style="margin: 5px 0; color: #444;">{result['description']}</p>
            </div>
            """
        
        results_html += "</div>"
        
        # Statistics
        avg_similarity = np.mean([r['similarity_score'] for r in results]) * 100
        stats = f"""
        **Search Statistics:**
        - Found: {len(results)} products
        - Average similarity: {avg_similarity:.1f}%
        - Best match: {results[0]['similarity_score']*100:.1f}%
        """
        
        return results_html, stats
    
    def get_search_analytics(self, query: str, results: List[Dict]) -> str:
        """
        Generate search analytics
        """
        if not results:
            return "No data for analysis"
        
        # Category analysis
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['similarity_score'])
        
        # Price analysis
        prices = [r['price'] for r in results]
        
        analysis = f"""
        **Results Analysis for: "{query}"**
        
        **Category Distribution:**
        """
        
        for cat, scores in categories.items():
            avg_score = np.mean(scores) * 100
            analysis += f"- {cat.title()}: {len(scores)} products (avg. similarity: {avg_score:.1f}%)\n"
        
        analysis += f"""
        
        **Price Analysis:**
        - Price range: ${min(prices)} - ${max(prices)}
        - Average price: ${np.mean(prices):.0f}
        - Median price: ${np.median(prices):.0f}
        
        **Result Quality:**
        - High similarity (>80%): {sum(1 for r in results if r['similarity_score'] > 0.8)} products
        - Medium similarity (60-80%): {sum(1 for r in results if 0.6 < r['similarity_score'] <= 0.8)} products
        - Low similarity (<60%): {sum(1 for r in results if r['similarity_score'] <= 0.6)} products
        """
        
        return analysis

def create_gradio_interface():
    """
    Create Gradio interface for search engine
    """
    print("Creating Gradio interface...")
    
    # Initialize search engine
    engine = MultimodalSearchEngine()
    
    # Create dataset and build index
    engine.create_sample_dataset(num_products=2000)
    engine.create_dummy_image_embeddings()
    engine.build_vector_database()
    
    def text_search_handler(query: str, num_results: int):
        """Handler for text search"""
        if not query.strip():
            return "Enter text query", "", ""
        
        try:
            results = engine.search_by_text(query, k=num_results)
            results_html, stats = engine.format_results_for_gradio(results)
            analysis = engine.get_search_analytics(query, results)
            return results_html, stats, analysis
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def image_search_handler(image, num_results: int):
        """Handler for image search"""
        if image is None:
            return "Upload image for search", "", ""
        
        try:
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')
            
            results = engine.search_by_image(image, k=num_results)
            results_html, stats = engine.format_results_for_gradio(results)
            analysis = engine.get_search_analytics("uploaded image", results)
            return results_html, stats, analysis
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    # Interface definition
    with gr.Blocks(title="Multimodal Search Engine", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>Multimodal Search Engine</h1>
            <p style="font-size: 18px; color: #666; margin: 10px 0;">
                Advanced product search system using CLIP and RAG
            </p>
            <p style="color: #888;">
                Search products using text descriptions or uploading images
            </p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Text search
            with gr.Tab("Text Search"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_query = gr.Textbox(
                            label="Text Query",
                            placeholder="e.g., 'black leather shoes', 'blue cotton t-shirt', 'wireless headphones'...",
                            lines=2
                        )
                        num_results_text = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Number of Results"
                        )
                        search_text_btn = gr.Button("Search", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Examples(
                            examples=[
                                ["black leather shoes"],
                                ["blue cotton t-shirt"],
                                ["wireless headphones"],
                                ["red dress"],
                                ["Nike sneakers"],
                                ["luxury watch"],
                                ["denim jacket"]
                            ],
                            inputs=[text_query],
                            label="Example Queries"
                        )
                
                text_results = gr.HTML(label="Results")
                
                with gr.Row():
                    text_stats = gr.Markdown(label="Statistics")
                    text_analysis = gr.Markdown(label="Analysis")
            
            # Tab 2: Image search
            with gr.Tab("Image Search"):
                with gr.Row():
                    with gr.Column():
                        image_query = gr.Image(
                            label="Upload Product Image",
                            type="pil",
                            height=300
                        )
                        num_results_image = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Number of Results"
                        )
                        search_image_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                
                image_results = gr.HTML(label="Results")
                
                with gr.Row():
                    image_stats = gr.Markdown(label="Statistics")
                    image_analysis = gr.Markdown(label="Analysis")
            
            # Tab 3: System info
            with gr.Tab("System Information"):
                gr.Markdown(f"""
                ## System Specification
                
                **Model:** OpenAI CLIP ViT-B/32  
                **Device:** {engine.device}  
                **Embedding Dimension:** {engine.embedding_dim}  
                **Product Count:** {len(engine.products_data):,}  
                **Vector Database:** FAISS  
                
                ## Technical Components
                
                - **CLIP (Contrastive Language-Image Pre-training)**: Multimodal model combining text/images
                - **FAISS**: Library for efficient similarity search in vector databases
                - **RAG (Retrieval-Augmented Generation)**: Architecture combining search with context generation
                - **MPS/CUDA**: GPU acceleration for faster processing
                
                ## Similarity Metrics
                
                System uses **cosine similarity** between embeddings:  
                - **> 80%**: Very high similarity  
                - **60-80%**: Medium similarity  
                - **< 60%**: Low similarity  
                
                ## Usage Tips
                
                **Text Search:**
                - Use specific descriptions (color, material, type)
                - Combine product features e.g., "black leather boots"
                - Experiment with brand names
                
                **Image Search:**
                - Use clear product images
                - Best results with plain backgrounds
                - System recognizes shapes, colors and textures
                """)
        
        # Connect handlers
        search_text_btn.click(
            text_search_handler,
            inputs=[text_query, num_results_text],
            outputs=[text_results, text_stats, text_analysis]
        )
        
        search_image_btn.click(
            image_search_handler,
            inputs=[image_query, num_results_image],
            outputs=[image_results, image_stats, image_analysis]
        )
    
    print("Gradio interface created!")
    return demo

if __name__ == "__main__":
    # Launch application
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )