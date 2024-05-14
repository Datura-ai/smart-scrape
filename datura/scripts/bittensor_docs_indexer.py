import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from llama_index.core import Settings
from pinecone import Pinecone, PodSpec
from llama_index.core import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import FlatReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore


class DocsIndexer:
    def __init__(self):
        self.OPENAI_APIKEY = os.environ.get("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        self.pinecone = Pinecone(api_key=self.PINECONE_API_KEY)
        Settings.embed_model = OpenAIEmbedding(
            api_key=self.OPENAI_APIKEY, model="text-embedding-3-large", embed_batch_size=100
        )
        self.docs_url = "https://github.com/opentensor/developer-docs"
        self.docs_path = f"{os.path.abspath(os.getcwd())}/bittensor-docs/docs"
        self.index_name = "documentation"
        self.subnets = {
            "alpha-1": "https://github.com/opentensor/prompting",
            "beta-2": "https://github.com/inference-labs-inc/omron-subnet",
            "gamma-3": "https://github.com/myshell-ai/MyShell-TTS-Subnet",
            "delta-4": "https://github.com/manifold-inc/targon",
            "epsilon-5": "https://github.com/OpenKaito/openkaito",
            "zeta-6": "https://github.com/NousResearch/finetuning-subnet",
            "eta-7": "https://github.com/eclipsevortex/SubVortex",
            "theta-8": "https://github.com/taoshidev/proprietary-trading-network",
            "iota-9": "https://github.com/RaoFoundation/pretraining",
            "kappa-10": "https://github.com/apollozkp/zkp-subnet",
            "lambda-11": "https://github.com/Cazure8/transcription-subnet",
            "mu-12": "https://github.com/backend-developers-ltd/ComputeHorde",
            "nu-13": "https://github.com/RusticLuftig/data-universe",
            "xi-14": "https://github.com/ceterum1/llm-defender-subnet",
            "omicron-15": "https://github.com/blockchain-insights/blockchain-data-subnet",
            "pi-16": "https://github.com/eseckft/BitAds.ai",
            "rho-17": "https://github.com/404-Repo/three-gen-subnet",
            "sigma-18": "https://github.com/corcel-api/cortex.t",
            "tau-19": "https://github.com/namoray/vision",
            "upsilon-20": "https://github.com/RogueTensor/bitagent_subnet",
            "phi-21": "https://github.com/ifrit98/storage-subnet",
            "chi-22": "https://github.com/surcyf123/smart-scrape",
            "psi-23": "https://github.com/NicheTensor/NicheImage",
            "omega-24": "https://github.com/omegalabsinc/omegalabs-bittensor-subnet",
            "alef-25": "https://github.com/bit-current/DistributedTraining",
            "bet-26": "https://github.com/Supreme-Emperor-Wang/ImageAlchemy",
            "gimel-27": "https://github.com/neuralinternet/compute-subnet",
            "dalet-28": "https://github.com/teast21/snpOracle",
            "he-29": "https://github.com/fractal-net/fractal",
            "wav-30": "https://github.com/womboai/wombo-bittensor-subnet",
            "zayin-31": "https://github.com/nimaaghli/NASChain",
            "chet-32": "https://github.com/It-s-AI/llm-detection"
        }

    def normalize_link(self, link: str) -> str:
        normalized = ''
        parts = link.split('/')

        slashed = 0
        for part in parts:
            if part == '' and slashed > 0:
                continue

            if part == '' and slashed == 0:
                slashed += 1
                normalized += f'{part}/'
                continue

            normalized += f'{part}/'

        return normalized

    def convert_link(self, match, file_path):
        link_text = match.group(1)
        link_target = match.group(2)

        if link_target.startswith('http'):
            return f"[{link_text}]({link_target})"

        base_url = "https://docs.bittensor.com"
        file_name = os.path.basename(file_path)
        folder = file_path.split(
            'bittensor-docs')[1] if 'bittensor-docs' in file_path else file_path
        folder = folder.replace(file_name, '')
        folder = folder.strip('/').replace('//', '/')

        if link_target.startswith('#'):
            _, file_name = os.path.split(file_path)
            file_name = os.path.splitext(file_name)[0]
            link = f"{base_url}/{folder}/{file_name}{link_target}"
            link = f"[{link_text}]({link})"
            return self.normalize_link(link.replace('.md', ''))

        if '#' in link_target:
            target = f"{folder}/{link_target}"
            link = f"{base_url}/{target.replace('.md', '')}"
            return self.normalize_link(f"[{link_text}]({link})")

        link_target = link_target.lstrip('./')
        if link_target.endswith('.md'):
            target = f"{folder}/{link_target}"
            link = f"{base_url}/{target.replace('.md', '')}"
            return self.normalize_link(f"[{link_text}]({link})")

        link_parts = link_target.split('/')
        link_file = '/'.join(link_parts[:-1])
        link_anchor = link_parts[-1].replace('.md', '')
        link = f"{base_url}/{link_file}/{link_anchor}"
        return self.normalize_link(f"[{link_text}]({link})")

    def fetch_docs(self):
        repo_dir = 'bittensor-docs'
        docs_path = os.path.join(repo_dir, 'docs')
        try:
            subprocess.run(
                ["git", "clone", self.docs_url, repo_dir], check=True)

            # Remove all directories and files except 'docs'
            for root, dirs, files in os.walk(repo_dir, topdown=False):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if dir_path != docs_path:
                        shutil.rmtree(dir_path)
                for file in files:
                    file_path = os.path.join(root, file)
                    if root != docs_path:
                        os.remove(file_path)

        except subprocess.CalledProcessError:
            print(f"Failed to clone {self.docs_url}")

    def read_docs(self):
        docs = []
        for root, _, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    document = FlatReader().load_data(Path(file_path))
                    for doc in document:
                        text = doc.text
                        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                        text = re.sub(pattern, lambda match: self.convert_link(
                            match, file_path), text)
                        doc.text = text
                        metadata = {k: v for k, v in doc.metadata.items()}
                        metadata['filepath'] = file_path
                        doc.metadata = metadata
                    docs.extend(document)

        path = self.docs_path.replace('docs/', '')
        shutil.rmtree(path)
        return docs

    def generate_link(self, file_extension, file_path):
        if file_extension is None or file_path is None:
            return None

        base_url = "https://docs.bittensor.com"
        location = file_path.split(
            'bittensor-docs')[1] if 'bittensor-docs' in file_path else file_path
        location = location.replace(file_extension, '')
        location = location.replace('//', '/')
        if location.endswith('index'):
            location = location[:-5]

        return f'{base_url}{location}'

    def convert_docs_to_nodes(self, docs):
        if len(docs) == 0:
            return []

        nodes = []

        for _, doc in enumerate(docs):
            metadata = {
                "title": doc.metadata.get("filename").replace('.md', '').capitalize(),
                "link": self.generate_link(doc.metadata.get('extension'), doc.metadata.get('filepath')),
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}

            if len(doc.text) <= 2000:
                node = TextNode(text=doc.text, metadata=metadata)
                nodes.append(node)
            else:
                chunks = []
                current_chunk = ""
                for line in doc.text.splitlines(keepends=True):
                    if line.startswith("##"):
                        if len(current_chunk) + len(line) > 2000:
                            chunks.append(current_chunk)
                            current_chunk = line
                        else:
                            current_chunk += line
                    else:
                        if len(current_chunk) + len(line) > 2000:
                            chunks.append(current_chunk)
                            current_chunk = line
                        else:
                            current_chunk += line
                if current_chunk:
                    chunks.append(current_chunk)

                for chunk in chunks:
                    node = TextNode(text=chunk, metadata=metadata)
                    nodes.append(node)

        return nodes

    def create_docs_index(
        self,
        ignore_creating_docs_index=False,
        ignore_fetching_docs=False,
    ):
        if not ignore_fetching_docs:
            print('fetching')
            self.fetch_docs()

        docs = self.read_docs()
        nodes = self.convert_docs_to_nodes(docs)

        if not ignore_creating_docs_index:
            self.pinecone.delete_index(self.index_name)
            self.pinecone.create_index(
                self.index_name,
                dimension=3072,
                metric="cosine",
                spec=PodSpec(
                    environment="us-east-1-aws",
                    pod_type="p1.x1",
                )
            )

        pinecone_index = self.pinecone.Index(self.index_name)
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            api_key=self.PINECONE_API_KEY,
            namespace="bittensor-documentation",
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )

        index_path = f"{os.path.abspath(os.getcwd())}/llama_index"
        index.set_index_id(self.index_name)
        index.storage_context.persist(persist_dir=f"{index_path}/index")

    def parse_otf_weights(self):
        nodes = {}

        file_path = os.path.join(os.path.abspath(os.getcwd()), 'otf.json')
        with open(file_path, 'r') as file:
            prenodes = json.load(file)

        for prenode in prenodes:
            node = ""
            title = ""
            subnet = ""
            for key, value in prenode.items():
                if value is None or value == "":
                    continue

                if key == "SN Name and SN Number":
                    title = value

                if key == "subnet":
                    subnet = value

                node += f'# {key.replace(":", "")}\n'
                node += f'{value}\n\n'

            current_nodes = nodes.get(subnet, [])
            current_nodes.append(
                TextNode(text=node, metadata={'title': title}))
            nodes[subnet] = current_nodes

        return nodes

    def generate_subnet_link(self, file_extension, file_path):
        if file_extension is None or file_path is None:
            return None

        subnet_name = file_path.split("/")[1]
        base_url = f'{self.subnets.get(subnet_name)}/blob/main/'
        current = file_path.replace(f'subnets/{subnet_name}/', '')
        return f'{base_url}{current}'

    def convert_subnet_link(self, match, file_path):
        link_text = match.group(1)
        link_target = match.group(2)

        if link_target.startswith('http'):
            return f"[{link_text}]({link_target})"

        subnet_name = file_path.split("/")[1]
        base_url = f'{self.subnets.get(subnet_name)}/blob/main/'

        if link_target.startswith('#'):
            _, file_name = os.path.split(file_path)
            file_name = os.path.splitext(file_name)[0]
            current = file_path.replace(f'subnets/{subnet_name}/', '')
            return self.normalize_link(f"[{link_text}]({base_url}{current}{link_target})")

        if '#' in link_target:
            current = file_path.replace(f'subnets/{subnet_name}/', '')
            base = current.split('/')[0]
            return self.normalize_link(f"[{link_text}]({base_url}{base}/{link_target})")

        link_target = link_target.lstrip('./')
        return self.normalize_link(f"[{link_text}]({base_url}{link_target})")

    def fetch_subnet_docs(self):
        for subnet_name, url in self.subnets.items():
            repo_dir = subnet_name
            try:
                subprocess.run(["git", "clone", url, repo_dir], check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to clone {url}")
                continue
            dest_dir = os.path.join("subnets", subnet_name)
            os.makedirs(dest_dir, exist_ok=True)

            for item in ["docs", "requirements.txt", "min_compute.yml", "README.md"]:
                src_path = os.path.join(repo_dir, item)
                if os.path.exists(src_path):
                    if os.path.isdir(src_path):
                        dest_path = os.path.join(dest_dir, item)
                        shutil.copytree(src_path, dest_path)
                    else:
                        dest_path = os.path.join(
                            dest_dir, os.path.basename(item))
                        shutil.copy(src_path, dest_path)
            shutil.rmtree(repo_dir)

    def read_subnet_docs(self):
        ALLOWED_EXTENSIONS = [".txt", ".md", ".yml"]
        docs = {}
        subnet_folders = [f.path for f in os.scandir("subnets") if f.is_dir()]

        for subnet_folder in subnet_folders:
            folder_name = os.path.basename(subnet_folder)
            current_docs = []

            # Walk through the root of the subnet folder
            for root, _, files in os.walk(subnet_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    if ext.lower() not in ALLOWED_EXTENSIONS:
                        continue
                    document = FlatReader().load_data(Path(file_path))
                    for doc in document:
                        text = doc.text
                        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                        text = re.sub(
                            pattern,
                            lambda match: self.convert_subnet_link(
                                match, file_path
                            ),
                            text
                        )
                        doc.text = text
                        metadata = {k: v for k, v in doc.metadata.items()}
                        metadata['filepath'] = file_path
                        doc.metadata = metadata
                        current_docs.append(doc)

            # Walk through the docs folder within the subnet folder
            docs_path = os.path.join(subnet_folder, "docs")
            if os.path.exists(docs_path):
                for root, _, files in os.walk(docs_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        _, ext = os.path.splitext(file)
                        if ext.lower() not in ALLOWED_EXTENSIONS:
                            continue
                        document = FlatReader().load_data(Path(file_path))
                        for doc in document:
                            text = doc.text
                            pattern = r'\[([^\]]+)\]\(([^)]+)\)'
                            text = re.sub(
                                pattern,
                                lambda match: self.convert_subnet_link(
                                    match, file_path
                                ),
                                text
                            )
                            doc.text = text
                            metadata = {k: v for k, v in doc.metadata.items()}
                            metadata['filepath'] = file_path
                            doc.metadata = metadata
                            current_docs.append(doc)

            docs[folder_name] = current_docs

        path = f'{os.path.abspath(os.getcwd())}/subnets'
        shutil.rmtree(path)
        return docs

    def convert_subnet_docs_to_nodes(self, docs):
        if len(docs) == 0:
            return {}

        nodes = {}

        for subnet, docs in docs.items():
            current_nodes = []

            for _, doc in enumerate(docs):
                metadata = {
                    "title": doc.metadata.get("filename").capitalize(),
                    "link": self.generate_subnet_link(doc.metadata.get('extension'), doc.metadata.get('filepath')),
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}

                if len(doc.text) <= 2000:
                    node = TextNode(text=doc.text, metadata=metadata)
                    current_nodes.append(node)
                else:
                    chunks = []
                    current_chunk = ""
                    for line in doc.text.splitlines(keepends=True):
                        if line.startswith("##"):
                            if len(current_chunk) + len(line) > 2000:
                                chunks.append(current_chunk)
                                current_chunk = line
                            else:
                                current_chunk += line
                        else:
                            if len(current_chunk) + len(line) > 2000:
                                chunks.append(current_chunk)
                                current_chunk = line
                            else:
                                current_chunk += line
                    if current_chunk:
                        chunks.append(current_chunk)

                    for chunk in chunks:
                        node = TextNode(text=chunk, metadata=metadata)
                        current_nodes.append(node)

                nodes[subnet] = current_nodes

        return nodes

    def create_subnet_docs_index(
        self,
        ignore_creating_subnets_index=True,
        ignore_fetching_subnet_docs=False
    ):
        try:
            if not ignore_fetching_subnet_docs:
                self.fetch_subnet_docs()

            docs = self.read_subnet_docs()
            nodes = self.convert_subnet_docs_to_nodes(docs)
            otf_weights = self.parse_otf_weights()

            if not ignore_creating_subnets_index:
                self.pinecone.create_index(
                    self.index_name,
                    dimension=3072,
                    metric="cosine",
                    spec=PodSpec(
                        environment="us-east-1-aws",
                        pod_type="p1.x1",
                    )
                )

            for subnet, nodes in nodes.items():
                print(f">>> {subnet} : {len(nodes)} nodes")
                merged = []
                merged.extend(nodes)
                if subnet in otf_weights:
                    merged.extend(otf_weights[subnet])
                    print(f"docs: >>> OTF: {subnet} : {len(otf_weights[subnet])} nodes")
                else:
                    print("docs: >>> OTF: <not-available>")

                pinecone_index = self.pinecone.Index(self.index_name)
                vector_store = PineconeVectorStore(
                    pinecone_index=pinecone_index,
                    api_key=self.PINECONE_API_KEY,
                    namespace=subnet
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store)
                index = VectorStoreIndex(
                    nodes=merged,
                    storage_context=storage_context,
                    show_progress=True,
                )

                index_path = f"{os.path.abspath(os.getcwd())}/llama_index"
                index.set_index_id(self.index_name)
                index.storage_context.persist(
                    persist_dir=f"{index_path}/index")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    def execute(
        self,
        ignore_creating_docs_index: bool,
        ignore_creating_subnets_index: bool,
        ignore_fetching_docs: bool,
        ignore_fetching_subnet_docs: bool,
    ):
        self.create_docs_index(
            ignore_creating_docs_index,
            ignore_fetching_docs
        )

        self.create_subnet_docs_index(
            ignore_creating_subnets_index,
            ignore_fetching_subnet_docs,
        )


if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print(
            "Usage: python3 datura/scripts/bittensor_docs_indexer.py [--ignore-creating-docs-index] [--ignore-creating-subnets-index] [--ignore-fetching-docs] [--ignore-fetching-subnet-docs]")
        sys.exit(1)

    ignore_creating_docs_index = "--ignore-creating-docs-index" in sys.argv or False
    ignore_creating_subnets_index = "--ignore-creating-subnets-index" in sys.argv or True
    ignore_fetching_docs = "--ignore-fetching-docs" in sys.argv or False
    ignore_fetching_subnet_docs = "--ignore-fetching-subnet-docs" in sys.argv or False

    docs_indexer = DocsIndexer()
    docs_indexer.execute(
        ignore_creating_docs_index,
        ignore_creating_subnets_index,
        ignore_fetching_docs,
        ignore_fetching_subnet_docs,
    )
