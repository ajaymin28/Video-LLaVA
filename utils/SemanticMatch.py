import torch
import transformers
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, T5EncoderModel
from tqdm import tqdm


class SemanticMatchBase(object):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.galleries = {}
    
    def register_new_gallery(self, gallery_name, gallery_data):
        if gallery_name in self.galleries.keys():
            raise ValueError("gallery with same name already exist")
        else:
            self.galleries[gallery_name] = {
                "entities": gallery_data,
                "embeddings": self.get_gallery_embeddings(gallery=gallery_data)
            }

    def get_gallery_embeddings(self, gallery, embedding_func):
        gallery_embeddings = []
        for text in tqdm(gallery, total=len(gallery)):
            gallery_embeddings.append(embedding_func(text))
        gallery_embeddings = torch.cat(gallery_embeddings, dim=0).to(self.device)
        return gallery_embeddings
    
    def get_embedding(self, text):
        return NotImplementedError()
    
    def getScoreForPair(self,text1,text2):
        text1_embedding = self.get_embedding(text1).to(self.device)
        text2_embedding = self.get_embedding(text2).to(self.device)
        return torch.cosine_similarity(text1_embedding,text2_embedding)
    
    def register_new_gallery(self, gallery_name, gallery_data):
        if gallery_name in self.galleries.keys():
            raise ValueError("gallery with same name already exist")
        else:
            self.galleries[gallery_name] = {
                "entities": gallery_data,
                "embeddings": self.get_gallery_embeddings(gallery=gallery_data,embedding_func=self.get_embedding)
            }
    
    def get_topk_matches(self, query, gallery_name, k=1):
        if gallery_name not in self.galleries.keys():
            raise ValueError("{} not registere, please call register_new_gallery(gallery_name, gallery_data) to register new gallery")
        
        query_embedding = self.get_embedding(query).to(self.device)
        similarity = torch.cosine_similarity(query_embedding, self.galleries[gallery_name]["embeddings"])
        topk_similarity, topk_indices = torch.topk(similarity, k)
        return [(self.galleries[gallery_name]["entities"][i], round(topk_similarity[idx].item(),3)) for idx,i in enumerate(topk_indices)]


class T5ForSoftMatching(SemanticMatchBase):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        self.model = T5EncoderModel.from_pretrained("google-t5/t5-base")
        self.model.eval()
        self.model = self.model.to(self.device)
        self.galleries = {}

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)  # Batch size 1
        outputs = self.model(input_ids=inputs)
        return outputs.last_hidden_state.mean(dim=1)

    # def get_topk_matches(self, query, gallery_name, k=1):
    #     if gallery_name not in self.galleries.keys():
    #         raise ValueError("{} not registere, please call register_new_gallery(gallery_name, gallery_data) to register new gallery")
    #     query_embedding = self.get_embedding(query).to(self.device)
    #     similarity = torch.cosine_similarity(query_embedding, self.galleries[gallery_name]["embeddings"])
    #     topk_similarity, topk_indices = torch.topk(similarity, k)
    #     return [(self.galleries[gallery_name]["entities"][i], round(topk_similarity[idx].item(),3)) for idx,i in enumerate(topk_indices)]


class BertForSoftMatching(SemanticMatchBase):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.model = self.model.to(self.device)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    # def get_topk_matches(self, query, k=1):
    #     query_embedding = self.get_embedding(query).to(self.device)
    #     similarity = torch.cosine_similarity(query_embedding, self.gallery_embeddings)
    #     topk_similarity, topk_indices = torch.topk(similarity, k)
    #     return [(self.gallery[i], round(topk_similarity[idx].item(),3)) for idx,i in enumerate(topk_indices)]