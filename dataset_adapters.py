import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import numpy as np
from PIL import Image
import pandas as pd
import os
import requests
from pathlib import Path
import zipfile
import json
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

class DatasetAdapter:
    """
    Base class for dataset adapters
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.products_data = []
        self.images = []
    
    def load_dataset(self):
        """Abstract method to load dataset"""
        raise NotImplementedError
    
    def get_products_data(self) -> List[Dict]:
        """Returns product data in standard format"""
        return self.products_data
    
    def get_images(self) -> List[Image.Image]:
        """Returns list of PIL images"""
        return self.images
    
    def save_processed_data(self, filename: str = "processed_dataset.json"):
        """Saves processed dataset"""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.products_data, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved: {filepath}")

class FashionMNISTAdapter(DatasetAdapter):
    """
    Adapter for Fashion-MNIST dataset
    """
    
    # Fashion-MNIST label mapping to categories and descriptions
    FASHION_MNIST_LABELS = {
        0: {"category": "clothing", "item_type": "t-shirt", "name": "T-shirt/top"},
        1: {"category": "clothing", "item_type": "trousers", "name": "Trouser"},
        2: {"category": "clothing", "item_type": "pullover", "name": "Pullover"},
        3: {"category": "clothing", "item_type": "dress", "name": "Dress"},
        4: {"category": "clothing", "item_type": "coat", "name": "Coat"},
        5: {"category": "shoes", "item_type": "sandal", "name": "Sandal"},
        6: {"category": "clothing", "item_type": "shirt", "name": "Shirt"},
        7: {"category": "shoes", "item_type": "sneaker", "name": "Sneaker"},
        8: {"category": "accessories", "item_type": "bag", "name": "Bag"},
        9: {"category": "shoes", "item_type": "ankle_boot", "name": "Ankle boot"}
    }
    
    COLORS = ["black", "white", "gray", "navy", "brown", "beige", "dark", "light"]
    BRANDS = ["Fashion Co.", "Style Plus", "Urban Wear", "Classic Style", "Modern Look", 
              "Trend Setter", "Fashion Forward", "Style Icon"]
    MATERIALS = {
        "clothing": ["cotton", "polyester", "wool", "silk", "linen", "denim"],
        "shoes": ["leather", "canvas", "synthetic", "suede", "rubber"],
        "accessories": ["leather", "canvas", "nylon", "synthetic"]
    }
    
    def __init__(self, data_dir: str = "./data", subset_size: Optional[int] = None):
        super().__init__(data_dir)
        self.subset_size = subset_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # CLIP requires 224x224
            transforms.Lambda(lambda x: x.convert('RGB'))  # Convert to RGB
        ])
    
    def load_dataset(self, split: str = "train") -> Tuple[List[Dict], List[Image.Image]]:
        """
        Loads Fashion-MNIST dataset
        
        Args:
            split: "train" or "test"
        """
        print(f"Loading Fashion-MNIST ({split} split)...")
        
        # Download dataset
        is_train = (split == "train")
        dataset = FashionMNIST(
            root=str(self.data_dir), 
            train=is_train, 
            download=True,
            transform=None  # We'll transform later
        )
        
        print(f"Loaded {len(dataset)} images")
        
        # Limit size if needed
        if self.subset_size:
            indices = np.random.choice(len(dataset), min(self.subset_size, len(dataset)), replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"Limited to {len(dataset)} samples")
        
        products_data = []
        images = []
        
        print("Processing data...")
        for idx, (image, label) in enumerate(tqdm(dataset)):
            # Convert image
            if isinstance(image, torch.Tensor):
                image_pil = transforms.ToPILImage()(image).convert('RGB')
            else:
                image_pil = image.convert('RGB')
            
            # Resize to 224x224 for CLIP
            image_resized = image_pil.resize((224, 224))
            images.append(image_resized)
            
            # Product metadata
            label_info = self.FASHION_MNIST_LABELS[int(label)]
            color = np.random.choice(self.COLORS)
            brand = np.random.choice(self.BRANDS)
            material = np.random.choice(self.MATERIALS[label_info["category"]])
            price = np.random.randint(20, 200)
            
            # Generate description
            description = self._generate_description(label_info, color, material, brand, price)
            
            product = {
                "id": idx,
                "name": f"{brand} {color.title()} {label_info['name']}",
                "category": label_info["category"],
                "item_type": label_info["item_type"],
                "color": color,
                "material": material,
                "brand": brand,
                "price": price,
                "description": description,
                "original_label": int(label),
                "image_path": f"fashion_mnist_{split}_{idx}.jpg",  # Save path
                "source": "Fashion-MNIST"
            }
            
            products_data.append(product)
        
        self.products_data = products_data
        self.images = images
        
        print(f"Processed {len(products_data)} products")
        return products_data, images
    
    def _generate_description(self, label_info: Dict, color: str, material: str, 
                            brand: str, price: int) -> str:
        """Generates realistic product description"""
        
        base_descriptions = {
            "t-shirt": f"Comfortable {color} {material} t-shirt perfect for casual wear. Features classic fit and soft fabric.",
            "trousers": f"Stylish {color} {material} trousers with modern cut. Perfect for both casual and formal occasions.",
            "pullover": f"Cozy {color} {material} pullover to keep you warm. Classic design with comfortable fit.",
            "dress": f"Elegant {color} {material} dress suitable for various occasions. Flattering silhouette and quality fabric.",
            "coat": f"Warm {color} {material} coat for cold weather. Stylish design with excellent protection from elements.",
            "sandal": f"Comfortable {color} {material} sandals perfect for summer. Durable construction and stylish design.",
            "shirt": f"Classic {color} {material} shirt for professional and casual wear. Quality fabric and timeless design.",
            "sneaker": f"Comfortable {color} {material} sneakers for everyday wear. Great support and modern style.",
            "bag": f"Practical {color} {material} bag for daily use. Spacious interior and durable construction.",
            "ankle_boot": f"Stylish {color} {material} ankle boots for all seasons. Comfortable fit and trendy design."
        }
        
        base_desc = base_descriptions.get(label_info["item_type"], f"Quality {color} {label_info['name'].lower()}")
        
        return f"{base_desc} Brand: {brand}. Material: {material.title()}. Price: ${price}. High quality and comfortable fit guaranteed."
    
    def save_images(self, save_dir: str = "fashion_mnist_images"):
        """Saves images to disk"""
        save_path = self.data_dir / save_dir
        save_path.mkdir(exist_ok=True)
        
        print(f"Saving {len(self.images)} images...")
        
        for idx, (image, product) in enumerate(tqdm(zip(self.images, self.products_data))):
            filename = f"{product['source'].lower().replace('-', '_')}_{idx:05d}.jpg"
            filepath = save_path / filename
            image.save(filepath, "JPEG", quality=95)
            
            # Update path in data
            self.products_data[idx]["image_path"] = str(filepath)
        
        print(f"Images saved at: {save_path}")

class DeepFashionAdapter(DatasetAdapter):
    """
    Adapter for DeepFashion dataset
    """
    
    DEEPFASHION_URL = "http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html"
    
    def __init__(self, data_dir: str = "./data", deepfashion_path: Optional[str] = None):
        super().__init__(data_dir)
        self.deepfashion_path = Path(deepfashion_path) if deepfashion_path else Path(data_dir) / "DeepFashion"
    
    def load_dataset(self, split: str = "train") -> Tuple[List[Dict], List[Image.Image]]:
        """
        Loads DeepFashion dataset
        
        Args:
            split: "train" or "test"
        """
        if not self.deepfashion_path.exists():
            print("DeepFashion dataset not found!")
            print(f"Download dataset from: {self.DEEPFASHION_URL}")
            print("Place it in: data/DeepFashion")
            return [], []
        
        print("Loading DeepFashion dataset...")
        
        # Define paths
        img_dir = self.deepfashion_path / "img"
        ann_dir = self.deepfashion_path / "Anno"
        list_file = self.deepfashion_path / "Eval" / f"list_eval_partition.txt"
        
        # Verify structure
        if not img_dir.exists():
            print(f"Missing 'img' folder: {img_dir}")
            return [], []
        if not list_file.exists():
            print(f"Missing partition file: {list_file}")
            return [], []
        
        # Read partition file
        print("Reading partition file...")
        with open(list_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip header lines
        
        products_data = []
        images = []
        product_id = 0
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png'}
        
        print("Processing images and annotations...")
        for line in tqdm(lines):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            img_path = parts[0]
            category = img_path.split('/')[0]
            split_flag = parts[2]  # 0=train, 1=val, 2=test
            
            # Filter by requested split
            if (split == "train" and split_flag != "0") or \
               (split == "test" and split_flag != "2"):
                continue
            
            full_img_path = img_dir / img_path
            
            # Check if image exists
            if not full_img_path.exists() or full_img_path.suffix.lower() not in image_extensions:
                continue
            
            try:
                # Load and resize image
                image = Image.open(full_img_path).convert('RGB')
                image_resized = image.resize((224, 224))
                images.append(image_resized)
                
                # Try to get attributes
                attr_file = ann_dir / "list_attr_img.txt"
                attributes = self._get_attributes(attr_file, img_path)
                
                # Create product metadata
                product = {
                    "id": product_id,
                    "name": img_path.split('/')[-1].split('.')[0].replace('_', ' ').title(),
                    "category": category,
                    "description": self._generate_description(category, attributes),
                    "image_path": str(full_img_path),
                    "source": "DeepFashion",
                    "split": split,
                    "attributes": attributes
                }
                
                # Add price if available in attributes
                if "price" in attributes:
                    product["price"] = attributes["price"]
                else:
                    product["price"] = np.random.randint(20, 500)
                
                products_data.append(product)
                product_id += 1
                
            except Exception as e:
                print(f"Error processing {full_img_path}: {e}")
                continue
        
        self.products_data = products_data
        self.images = images
        
        print(f"Loaded {len(products_data)} products from DeepFashion ({split} split)")
        return products_data, images
    
    def _get_attributes(self, attr_file: Path, img_path: str) -> Dict:
        """Reads attributes for an image from the attribute file"""
        attributes = {}
        
        if not attr_file.exists():
            return attributes
            
        try:
            with open(attr_file, 'r') as f:
                lines = f.readlines()[2:]  # Skip header lines
                
            # Find the line for this image
            for line in lines:
                parts = line.strip().split()
                if parts[0] == img_path:
                    # The next 1000 parts are attribute flags
                    attributes = {
                        "texture": "smooth" if parts[1] == "1" else "textured",
                        "fabric": "stretchy" if parts[2] == "1" else "non-stretchy",
                        "shape": "fitted" if parts[3] == "1" else "loose",
                        "part": "full-body" if parts[4] == "1" else "partial",
                        "style": "casual" if parts[5] == "1" else "formal"
                    }
                    break
        except Exception as e:
            print(f"Error reading attributes: {e}")
        
        return attributes
    
    def _generate_description(self, category: str, attributes: Dict) -> str:
        """Generates product description from attributes"""
        description = f"{category.replace('_', ' ').title()} from DeepFashion dataset. "
        
        if attributes:
            description += "Features: "
            desc_parts = []
            
            if "texture" in attributes:
                desc_parts.append(f"{attributes['texture']} texture")
            if "fabric" in attributes:
                desc_parts.append(f"{attributes['fabric']} fabric")
            if "shape" in attributes:
                desc_parts.append(f"{attributes['shape']} fit")
            if "style" in attributes:
                desc_parts.append(f"{attributes['style']} style")
            
            description += ", ".join(desc_parts)
        else:
            description += "High-quality fashion item."
        
        if "price" in attributes:
            description += f" Price: ${attributes['price']}."
        else:
            description += f" Price: ${np.random.randint(20, 500)}."
        
        return description

class CustomDatasetAdapter(DatasetAdapter):
    """
    Adapter for custom datasets
    """
    
    def __init__(self, data_dir: str = "./data", 
                 images_dir: Optional[str] = None,
                 metadata_file: Optional[str] = None):
        super().__init__(data_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.metadata_file = Path(metadata_file) if metadata_file else None
    
    def load_from_folder(self, images_dir: str, 
                        categories_mapping: Optional[Dict] = None) -> Tuple[List[Dict], List[Image.Image]]:
        """
        Loads dataset from image folder
        
        Folder structure:
        images_dir/
        ├── category1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── category2/
            ├── image3.jpg
            └── image4.jpg
        """
        images_path = Path(images_dir)
        if not images_path.exists():
            raise ValueError(f"Folder {images_dir} does not exist!")
        
        print(f"Loading images from: {images_path}")
        
        products_data = []
        images = []
        product_id = 0
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for category_dir in images_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            print(f"Processing category: {category}")
            
            for image_file in category_dir.iterdir():
                if image_file.suffix.lower() not in image_extensions:
                    continue
                
                try:
                    # Load image
                    image = Image.open(image_file).convert('RGB')
                    image_resized = image.resize((224, 224))
                    images.append(image_resized)
                    
                    # Product metadata
                    product_name = image_file.stem.replace('_', ' ').title()
                    
                    product = {
                        "id": product_id,
                        "name": product_name,
                        "category": category,
                        "item_type": category,  # Can be customized
                        "description": f"Product from category {category}. High quality item.",
                        "image_path": str(image_file),
                        "source": "Custom Dataset"
                    }
                    
                    # Add category mapping if provided
                    if categories_mapping and category in categories_mapping:
                        product.update(categories_mapping[category])
                    
                    products_data.append(product)
                    product_id += 1
                    
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
                    continue
        
        self.products_data = products_data
        self.images = images
        
        print(f"Loaded {len(products_data)} products from {len(set(p['category'] for p in products_data))} categories")
        return products_data, images
    
    def load_from_csv(self, csv_file: str, images_dir: str) -> Tuple[List[Dict], List[Image.Image]]:
        """
        Loads dataset from CSV metadata file
        
        CSV should contain columns:
        - image_filename: image file name
        - name: product name
        - category: category
        - description: description (optional)
        - price: price (optional)
        """
        csv_path = Path(csv_file)
        images_path = Path(images_dir)
        
        if not csv_path.exists():
            raise ValueError(f"CSV file {csv_file} does not exist!")
        if not images_path.exists():
            raise ValueError(f"Images folder {images_dir} does not exist!")
        
        print(f"Loading metadata from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        products_data = []
        images = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_filename = row['image_filename']
            image_path = images_path / image_filename
            
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            
            try:
                # Load image
                image = Image.open(image_path).convert('RGB')
                image_resized = image.resize((224, 224))
                images.append(image_resized)
                
                # Metadata from CSV
                product = {
                    "id": len(products_data),
                    "name": row.get('name', f"Product {idx}"),
                    "category": row.get('category', 'unknown'),
                    "description": row.get('description', 'No description available'),
                    "price": row.get('price', np.random.randint(20, 200)),
                    "image_path": str(image_path),
                    "source": "CSV Dataset"
                }
                
                # Add all additional columns
                for col in df.columns:
                    if col not in ['image_filename', 'name', 'category', 'description', 'price']:
                        product[col] = row[col]
                
                products_data.append(product)
                
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
        
        self.products_data = products_data
        self.images = images
        
        print(f"Loaded {len(products_data)} products from CSV")
        return products_data, images

class DatasetIntegrator:
    """
    Class for integrating dataset adapters with search engine
    """
    
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.current_adapter = None
    
    def load_fashion_mnist(self, subset_size: int = 2000, split: str = "train"):
        """Loads Fashion-MNIST into search engine"""
        print("Integrating Fashion-MNIST with Search Engine...")
        
        adapter = FashionMNISTAdapter(subset_size=subset_size)
        products_data, images = adapter.load_dataset(split=split)
        
        # Integrate with search engine
        self.search_engine.products_data = products_data
        self.search_engine.images = images
        
        # Generate CLIP embeddings
        print("Generating CLIP embeddings...")
        self._generate_clip_embeddings()
        
        # Build FAISS index
        print("Building FAISS index...")
        self.search_engine.build_vector_database()
        
        self.current_adapter = adapter
        print("Fashion-MNIST integrated with Search Engine!")
        
        return adapter
    
    def load_custom_dataset(self, images_dir: str, metadata_csv: Optional[str] = None,
                          categories_mapping: Optional[Dict] = None):
        """Loads custom dataset into search engine"""
        print("Integrating custom dataset with Search Engine...")
        
        adapter = CustomDatasetAdapter()
        
        if metadata_csv:
            products_data, images = adapter.load_from_csv(metadata_csv, images_dir)
        else:
            products_data, images = adapter.load_from_folder(images_dir, categories_mapping)
        
        # Integrate with search engine
        self.search_engine.products_data = products_data
        self.search_engine.images = images
        
        # Generate CLIP embeddings
        print("Generating CLIP embeddings...")
        self._generate_clip_embeddings()
        
        # Build FAISS index
        print("Building FAISS index...")
        self.search_engine.build_vector_database()
        
        self.current_adapter = adapter
        print("Custom dataset integrated with Search Engine!")
        
        return adapter
    
    def _generate_clip_embeddings(self):
        """Generates CLIP embeddings for loaded images"""
        if not hasattr(self.search_engine, 'images') or not self.search_engine.images:
            raise ValueError("No images to process!")
        
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(self.search_engine.images), batch_size)):
            batch_images = self.search_engine.images[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.search_engine.processor(
                    images=batch_images, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(self.search_engine.device) for k, v in inputs.items()}
                
                image_features = self.search_engine.model.get_image_features(**inputs)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                
                embeddings.append(image_features.cpu().numpy())
        
        self.search_engine.image_embeddings = np.vstack(embeddings)
        print(f"Generated {len(self.search_engine.image_embeddings)} embeddings")
    
    def get_dataset_info(self) -> Dict:
        """Returns info about currently loaded dataset"""
        if not self.current_adapter:
            return {"error": "No dataset loaded"}
        
        products = self.search_engine.products_data
        categories = [p['category'] for p in products]
        
        return {
            "total_products": len(products),
            "categories": list(set(categories)),
            "category_counts": pd.Series(categories).value_counts().to_dict(),
            "adapter_type": type(self.current_adapter).__name__,
            "has_images": len(self.search_engine.images) if hasattr(self.search_engine, 'images') else 0,
            "has_embeddings": len(self.search_engine.image_embeddings) if self.search_engine.image_embeddings is not None else 0
        }

# Usage example
def example_usage():
    """Example of integration with different datasets"""
    from multimodal_search_engine import MultimodalSearchEngine
    
    # Initialize search engine
    engine = MultimodalSearchEngine()
    integrator = DatasetIntegrator(engine)
    
    print("Available integration options:")
    print("1. Fashion-MNIST")
    print("2. Custom dataset from folder")
    print("3. Custom dataset from CSV")
    print("4. DeepFashion dataset")
    
    # Option 1: Fashion-MNIST
    print("\nExample 1: Fashion-MNIST")
    adapter = integrator.load_fashion_mnist(subset_size=1000)
    
    # Test search
    results = engine.search_by_text("black sneakers", k=3)
    print("Search results:", [r['name'] for r in results])
    
    # Dataset info
    info = integrator.get_dataset_info()
    print("Dataset info:", info)
    
if __name__ == "__main__":
    example_usage()