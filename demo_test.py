#!/usr/bin/env python3
"""
Demo script for Multimodal Search Engine
Demonstrates all key system functionalities
"""
import os
import sys
import time
from multimodal_search_engine import MultimodalSearchEngine
from visualization_utils import EmbeddingVisualizer, SearchEngineEvaluator, run_full_evaluation
import requests
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def print_header(title):
    """Displays a formatted header"""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)

def print_results(results, title="Search Results"):
    """Displays search results in readable format"""
    print(f"\n{title}:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        similarity_percent = result['similarity_score'] * 100
        print(f"{i}. {result['name']}")
        print(f"   Category: {result['category'].title()}")
        print(f"   Price: ${result['price']}")
        print(f"   Similarity: {similarity_percent:.1f}%")
        print(f"   Description: {result['description'][:80]}...")
        print()

def demo_basic_functionality():
    """Demonstrates core search engine functionality"""
    print_header("Basic Search Engine Functionality")
    
    # Initialization
    print("Initializing Search Engine...")
    engine = MultimodalSearchEngine()
    
    # Create dataset
    print("Creating sample dataset...")
    engine.create_sample_dataset(num_products=2000)
    
    # Create embeddings and index
    print("Generating embeddings...")
    engine.create_dummy_image_embeddings()
    
    print("Building vector database...")
    engine.build_vector_database()
    
    print("Search Engine ready for use!")
    
    return engine

def demo_text_search(engine):
    """Demonstrates text search functionality"""
    print_header("Text Search Demo")
    
    # Sample queries
    queries = [
        "black leather shoes",
        "blue cotton t-shirt", 
        "Nike sneakers",
        "luxury watch",
        "red dress",
        "wireless headphones"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        start_time = time.time()
        
        results = engine.search_by_text(query, k=3)
        
        search_time = time.time() - start_time
        print(f"Search time: {search_time:.3f}s")
        
        if results:
            print_results(results[:3])
            
            # Category analysis
            categories = [r['category'] for r in results]
            unique_categories = list(set(categories))
            print(f"Found categories: {', '.join(unique_categories)}")
        else:
            print("No results found")

def demo_image_search(engine):
    """Demonstrates real image search functionality"""
    print_header("Image Search Demo")
    
    # Define demo images directory
    demo_data_dir = "demo_data"
    demo_images = [
        "shoes.jpg",
        "tshirt.jpg",
        "headphones.jpg",
        "watch.jpg",
        "dress.jpg"
    ]
    
    # Create demo directory if it doesn't exist
    if not os.path.exists(demo_data_dir):
        os.makedirs(demo_data_dir)
        print(f"Created empty demo directory: {demo_data_dir}")
        print("Please add some product images to use this feature.")
        print("Continuing with simulated search...")
        return demo_image_search_simulated(engine)
    
    # Find available demo images
    available_images = [f for f in os.listdir(demo_data_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not available_images:
        print("No images found in demo_data directory")
        print("Please add some product images to use this feature.")
        print("Continuing with simulated search...")
        return demo_image_search_simulated(engine)
    
    print("Available demo images:")
    for i, img in enumerate(available_images, 1):
        print(f"{i}. {img}")
    
    # Select an image to search with
    try:
        selection = np.random.choice([i for i in range(0,5)])
        if selection < 0 or selection >= len(available_images):
            raise ValueError
        image_path = os.path.join(demo_data_dir, available_images[selection])
    except (ValueError, IndexError):
        print("Invalid selection. Using first available image.")
        image_path = os.path.join(demo_data_dir, available_images[0])
    
    # Load and display image info
    try:
        image = Image.open(image_path)
        print(f"\nSearching with: {os.path.basename(image_path)}")
        print(f"   Format: {image.format}")
        print(f"   Size: {image.size} pixels")
        print(f"   Mode: {image.mode}")
        
        # Show small preview if possible
        if image.width > 300 or image.height > 300:
            preview = image.copy()
            preview.thumbnail((200, 200))
            preview.show(title="Search Image Preview")
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using default image instead...")
        image = Image.new('RGB', (224, 224), color='gray')
    
    # Perform search
    start_time = time.time()
    results = engine.search_by_image(image, k=5)
    search_time = time.time() - start_time
    
    print(f"Search time: {search_time:.3f}s")
    print_results(results)
    
    # Show top result image
    if results:
        top_result = results[0]
        print(f"Top result: {top_result['name']} ({top_result['similarity_score']*100:.1f}% match)")
        
        # Try to show top result image
        try:
            response = requests.get(top_result['image_url'], stream=True, timeout=5)
            if response.status_code == 200:
                result_img = Image.open(BytesIO(response.content))
                result_img.show(title=f"Top Result: {top_result['name']}")
            else:
                print("Couldn't retrieve product image")
        except Exception as e:
            print(f"Error showing product image: {e}")

def demo_image_search_simulated(engine):
    """Fallback simulated image search"""
    print("\nUsing simulated image search...")
    
    # Select random product as "query image"
    random_product = np.random.choice(engine.products_data)
    print(f"   Simulating product image: {random_product['name']}")
    print(f"   Expected category: {random_product['category']}")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='white')
    
    start_time = time.time()
    results = engine.search_by_image(dummy_image, k=3)
    search_time = time.time() - start_time
    
    print(f"Search time: {search_time:.3f}s")
    print_results(results)
    
    # Check category matches
    matching_category = sum(1 for r in results if r['category'] == random_product['category'])
    print(f"Products in expected category: {matching_category}/{len(results)}")

def demo_analytics(engine):
    """Demonstrates analytics and visualization"""
    print_header("Analytics and Visualization")
    
    try:
        # Initialize visualizer
        print("Initializing visualization module...")
        visualizer = EmbeddingVisualizer(engine)
        
        # Dimensionality reduction
        print("Reducing embedding dimensions (t-SNE)...")
        visualizer.reduce_dimensions(method="tsne", sample_size=500)
        
        # Clustering
        print("Clustering products...")
        clusters = visualizer.cluster_embeddings(n_clusters=6)
        
        print(f"Created {len(set(clusters))} clusters")
        
        # Category distribution analysis
        analysis = visualizer.analyze_category_distribution()
        print("\nDataset analysis:")
        print(f"- Total products: {analysis['total_products']}")
        print("- Category distribution:")
        for cat, count in analysis['category_counts'].items():
            percentage = (count / analysis['total_products']) * 100
            print(f"  • {cat.title()}: {count} ({percentage:.1f}%)")
        
        # Visualization (info only, no display)
        print("\nGenerating visualizations...")
        print("- Embedding space plot")
        print("- Product category analysis") 
        print("- Similarity heatmap")
        print("Run the full Gradio interface to see visualizations!")
        
    except Exception as e:
        print(f"Visualization module error: {e}")
        print("Continuing without visualization...")

def demo_performance_evaluation(engine):
    """Demonstrates system performance evaluation"""
    print_header("System Performance Evaluation")
    
    try:
        # Initialize evaluator
        evaluator = SearchEngineEvaluator(engine)
        
        # Create test queries
        print("Generating test queries...")
        test_queries = evaluator.create_test_queries(n_queries=20)
        
        print(f"Created {len(test_queries)} test queries")
        print("\nSample queries:")
        for i, query in enumerate(test_queries[:5], 1):
            print(f"{i}. '{query['query']}' (expected category: {query['expected_category']})")
        
        # Evaluation
        print("\nStarting evaluation...")
        evaluation_results = evaluator.evaluate_search_quality(test_queries, k=5)
        
        # Display results
        print("\nEvaluation results:")
        print(f"- Precision@5: {evaluation_results['mean_precision_at_k']:.3f} ({evaluation_results['mean_precision_at_k']*100:.1f}%)")
        print(f"- Category Accuracy: {evaluation_results['mean_category_accuracy']:.3f} ({evaluation_results['mean_category_accuracy']*100:.1f}%)")
        print(f"- Average similarity: {evaluation_results['mean_avg_similarity']:.3f} ({evaluation_results['mean_avg_similarity']*100:.1f}%)")
        
        # Results interpretation
        precision = evaluation_results['mean_precision_at_k']
        accuracy = evaluation_results['mean_category_accuracy']
        
        print("\nInterpretation:")
        if precision > 0.8:
            print("Precision@5: Excellent result!")
        elif precision > 0.6:
            print("Precision@5: Good result")
        elif precision > 0.4:
            print("Precision@5: Average result")
        else:
            print("Precision@5: Needs improvement")
            
        if accuracy > 0.9:
            print("Category Accuracy: Excellent result!")
        elif accuracy > 0.7:
            print("Category Accuracy: Good result")
        elif accuracy > 0.5:
            print("Category Accuracy: Average result")
        else:
            print("Category Accuracy: Needs improvement")
            
    except Exception as e:
        print(f"Evaluation error: {e}")

def demo_advanced_queries(engine):
    """Demonstrates advanced query capabilities"""
    print_header("Advanced Queries and Use Cases")
    
    advanced_queries = [
        {
            "query": "luxury black leather",
            "description": "Multi-attribute query (luxury + color + material)"
        },
        {
            "query": "Nike",
            "description": "Brand-only query"
        },
        {
            "query": "running",
            "description": "Functional query"
        },
        {
            "query": "cheap headphones under 100",
            "description": "Price-limited query"
        },
        {
            "query": "summer casual wear",
            "description": "Contextual query"
        }
    ]
    
    for query_info in advanced_queries:
        query = query_info["query"]
        description = query_info["description"]
        
        print(f"\nTest: {description}")
        print(f"Query: '{query}'")
        
        results = engine.search_by_text(query, k=3)
        
        if results:
            # Results analysis
            categories = [r['category'] for r in results]
            prices = [r['price'] for r in results]
            similarities = [r['similarity_score'] for r in results]
            
            print(f"   Results analysis:")
            print(f"   Categories: {list(set(categories))}")
            print(f"   Price range: ${min(prices)}-${max(prices)}")
            print(f"   Average similarity: {np.mean(similarities):.3f}")
            
            # Top result
            best_result = results[0]
            print(f"Top result: {best_result['name']} ({best_result['similarity_score']*100:.1f}%)")
        else:
            print("No results found")

def demo_system_info(engine):
    """Displays system information"""
    print_header("System Information")
    
    print(f"Compute device: {engine.device}")
    print(f"CLIP model: {engine.model.config.name_or_path if hasattr(engine.model.config, 'name_or_path') else 'openai/clip-vit-base-patch32'}")
    print(f"Embedding dimension: {engine.embedding_dim}")
    print(f"Product count: {len(engine.products_data):,}")
    print(f"FAISS index size: {engine.faiss_index.ntotal if engine.faiss_index else 0:,}")
    
    # Dataset statistics
    categories = [p['category'] for p in engine.products_data]
    prices = [p['price'] for p in engine.products_data]
    
    print(f"\n Dataset statistics:")
    print(f"   Categories: {len(set(categories))}")
    print(f"   Average price: ${np.mean(prices):.2f}")
    print(f"   Price range: ${min(prices)} - ${max(prices)}")
    
    # Performance info
    if hasattr(engine, 'device'):
        if engine.device.type == 'mps':
            print("Acceleration: Apple Metal Performance Shaders (MPS)")
        elif engine.device.type == 'cuda':
            print("Acceleration: NVIDIA CUDA")
        else:
            print("Compute: CPU (no GPU acceleration)")

def run_interactive_demo():
    """Runs interactive demo mode"""
    print_header("Interactive Demo")
    
    print("Interactive mode - enter your own queries!")
    print("Available commands:")
    print("- 'quit' or 'exit' - end session")
    print("- 'stats' - show system stats")
    print("- 'help' - show help")
    print("- or enter a search query")
    
    engine = demo_basic_functionality()
    
    while True:
        print("\n" + "-"*40)
        user_input = input("Enter query: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Thanks for testing!")
            break
        elif user_input.lower() == 'stats':
            demo_system_info(engine)
        elif user_input.lower() == 'help':
            print("  Sample queries:")
            print("- black leather shoes")
            print("- Nike sneakers")
            print("- luxury watch")
            print("- blue cotton t-shirt")
        elif user_input:
            try:
                results = engine.search_by_text(user_input, k=3)
                print_results(results)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please enter a query or command")

def main():
    """Main demo function"""
    print("MULTIMODAL SEARCH ENGINE - COMPLETE DEMO")
    print("=" * 60)
    print("This script demonstrates all system functionalities")
    print("Run with '--interactive' parameter for interactive mode")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_demo()
        return
    
    # Full automated demo
    engine = demo_basic_functionality()
    
    # Feature demos
    demo_text_search(engine)
    demo_image_search(engine)
    demo_advanced_queries(engine)
    demo_analytics(engine)
    demo_performance_evaluation(engine)
    demo_system_info(engine)
    
    print_header("Demo Summary")
    print("   All tests completed successfully!")
    print("\n Next steps:")
    print("1. Run full interface: python multimodal_search_engine.py")
    print("2. Try interactive mode: python demo_test.py --interactive")
    print("3. Customize parameters for your needs")
    print("4. Add your own product data")
    
    print("\nUseful commands:")
    print("- python multimodal_search_engine.py  # Full Gradio application")
    print("- python demo_test.py --interactive   # Interactive testing")
    print("- python -c \"from visualization_utils import *; help()\"  # API help")

if __name__ == "__main__":
    main()