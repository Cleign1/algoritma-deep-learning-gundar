# PROGRAM STUDI INFORMATIKA  

UNIVERSITAS GUNADARMA  

**ALGORITMA DEEP LEARNING**  
**PERTEMUAN KE 7 – AGENT DAN ORKESTRASI LLM**

---

## **RPS (Rencana Pembelajaran Semester)**  

1. **Konsep Agent, Tools untuk Agent, LangChain, dan NVIDIA NIMS**  
2. **Teknik perbaikan dengan RAG**  
3. **Membangun Agen RAG dengan LLM**

---

## **1. Konsep Agent, Tools untuk Agent, LangChain, dan NVIDIA NIMS**  

## **Konsep Agent dalam AI**  

### Definisi Agent

- **Agent** adalah sistem yang dapat mengamati lingkungan, mengambil keputusan, dan melakukan tindakan untuk mencapai tujuan tertentu.  
- Bisa berupa fisik (robot) atau virtual (software).  

### Agent dalam LLM

- **LLM** LLM seringkali diimplementasikan sebagai agent yang dapat berinteraksi dengan pengguna secara natural.  
- Mereka dapat memahami pertanyaan, memberikan jawaban, dan bahkan melakukan tugas-tugas kompleks.

### Karakteristik Agent

- **Autonomi:** Beroperasi secara mandiri.  
- **Reaktif:** Merespons perubahan lingkungan.  
- **Tujuan:** Memiliki tujuan spesifik.  
- **Fleksibel:** Beradaptasi dengan situasi berbeda.

---

## **Tools untuk Membangun Agent**  

### LangChain

- Framework yang memudahkan pengembangan aplikasi AI yang berbasis pada model bahasa besar.
- Menyediakan modul-modul untuk:
  - Membuat prompt: Membentuk input yang tepat untuk model LLM.
  - Mengelola data: Mengorganisir dan mengakses data yang dibutuhkan oleh agent.
  - Membuat rantai tugas: Menghubungkan berbagai komponen agent menjadi sebuah alur kerja.

### Nvidia NeMo Megatron

- Framework untuk melatih dan menyempurnakan model bahasa besar skala besar  
- Menyediakan alat untuk:
  - Pelatihan terdistribusi: Melatih model pada cluster GPU.
  - Fine-tuning: Menyesuaikan model yang sudah dilatih pada tugas spesifik

### Hugging Face Transformers

- Library yang menyediakan berbagai model pre-trained dan tools untuk fine-tuning.
- Menyediakan model-model bahasa yang populer seperti BERT, GPT, dan banyak lagi.

---

## **2. Teknik perbaikan dengan RAG (Retrieval-Augmented Generation)**  

RAG adalah teknik yang menggabungkan kekuatan model bahasa besar dengan kemampuan untuk mengambil informasi dari sumber eksternal.

### Cara Kerja

1. **Query:** Pengguna mengajukan pertanyaan.  
2. **Retrieval:** Sistem mencari informasi yang relevan dari basis data.
3. **Generation:** Model bahasa besar menggunakan
informasi yang diambil untuk menghasilkan
jawaban yang komprehensif dan akurat.  

### Keuntungan RAG

- **Informasi yang lebih akurat**: Model dapat mengakses informasi terkini.
- **Jawaban yang lebih relevan**: Model dapat memberikan jawaban yang spesifik berdasarkanpertanyaan pengguna.
- **Kemampuan generalisasi yang lebih baik**: Model dapat menangani pertanyaan yang tidak pernah dilatih sebelumnya.

## Teknik Perbaikan

### Peningkatan Kualitas Data

- **Pembersihan Data**: Menghilangkan noise, inkonsistensi, dan data yang tidak relevan dalam basis data.
- **Augmentasi Data**: Menambah variasi data dengan teknik seperti synonymy, backtranslation, dan data augmentation.
- **Labeling Data**: Memberikan label yang akurat pada data untuk meningkatkan kinerja model dalam tugas-tugas seperti klasifikasi dan pengenalan entitas.

### Optimasi Model

- **Fine-tuning**: Menyesuaikan parameter model pada dataset yang lebih spesifik.
- **Regularisasi**: Mencegah overfitting dengan teknik seperti L1/L2 regularization, dropout.
- **Arsitektur Model**: Mencoba arsitektur model yang berbeda, seperti transformer yang lebih dalam atau menggunakan attention mechanism yang lebih kompleks.

### Perbaikan Retrieval

- **Peningkatan Relevance**: Menggunakan teknik pencarian informasi yang lebih canggih, seperti semantic search atau vector database.
- **Diversifikasi Hasil**: Menampilkan hasil yang beragam untuk menghindari bias.
- **Filtering**: Membuang hasil yang tidak relevan atau duplikat.

### Pengelolaan Context

- **Context Window**: Memperpanjang panjang konteks yang dapat diproses oleh model untuk memahami hubungan yang lebih kompleks antara kata-kata.
- **Contextual Embedding**: Menggunakan embedding yang sensitif terhadap konteks untuk mewakili kata-kata.

### Evaluasi yang Lebih Baik

- **Metrik yang Komprehensif**: Menggunakan berbagai metrik untuk mengevaluasi kinerja model, seperti BLEU, ROUGE, METEOR, dan human evaluation.
- **Analisis Kesalahan**: Menganalisis kesalahan yang dilakukan oleh model untuk mengidentifikasi area yang perlu perbaikan.

### Teknik Reinforcement Learning

- **Reward Model**: Mendesain reward model yang tepat untuk mendorong model menghasilkan jawaban yang berkualitas tinggi.
- **Policy Gradient**: Menggunakan algoritma reinforcement learning untuk memperbarui parameter model berdasarkan reward yang diperoleh.

---

## **3. Tugas Kelompok Membangun Agen RAG dengan LLM**

### **Langkah-langkah Membangun Agen RAG dengan LLM**  

1. **Pilih Model LLM**: Sesuai dengan tugas dan sumber daya.  
2. **Buat Basis Data**: Kumpulkan data yang relevan dan organisasikan dalam format yang dapat diakses oleh model
3. **Implementasi Retrieval**: Gunakan teknik pencarian informasi seperti vector database atau search engine untuk menemukan informasi yang relevan.
4. **Integrasi dengan LLM**: Gabungkan hasil pencarian dengan model LLM untuk menghasilkan jawaban.
5. **Evaluasi**: Evaluasi kinerja agent berdasarkan metrik yang relevan, seperti akurasi, relevansi, dan koherensi.

## **Membangun AI Agent menggunakan LangChain, Nvidia NeMo, dan Hugging Face Transformers**  

## Membangun AI Agent sederhana menggunakan LangChain, Nvidia NeMo Megatron, dan Hugging Face Transformers

### Inisialisasi LangChain

- LangChain menyediakan infrastruktur untuk membangun agen yang dapat memanfaatkan berbagai model dan alat NLP.
- LangChain dapat diintegrasikan dengan berbagai LLM (Large Language Models), seperti dari Hugging Face atau Nvidia NeMo.

### Integrasi Nvidia NeMo Megatron

- Nvidia NeMo Megatron adalah toolkit pembelajaran mendalam yang digunakan untuk melatih dan mengimplementasikan LLM berskala besar. Anda dapat memanfaatkan model pra-latihannya untuk tugas NLP.

### Integrasi Hugging Face Transformers

- Hugging Face Transformers menyediakan berbagai model NLP yang telah dilatih, seperti GPT, BERT, atau model lainnya. Ini digunakan untuk pemrosesan teks dan memahami konteks percakapan.

### Membangun Agen di LangChain

- Agent akan berinteraksi dengan pengguna menggunakan input bahasa alami, kemudian memberikan respon yang diproses menggunakan model dari NeMo Megatron dan Hugging Face Transformers.

## Langkah-langkah Pembuatan Program

### 1. Instalasi Pustaka

```bash
pip install langchain
pip install transformers
pip install nemo_toolkit['all']
pip install accelerate
pip install torch
```

### 2. Mengimpor Modul dan Inisiasi Model

```bash
from langchain import LLMPipeline
from transformers import pipeline
import nemo.collections.nlp as nemo_nlp

# Inisialisasi Model Hugging Face
hf_pipeline = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

# Inisialisasi Model Nvidia NeMo
nemo_model = nemo_nlp.models.megatron_gpt_model.MegatronGPTModel.from_pretrained(model_name="megatron-11b")

# Gabungkan Model
class SimpleAgent(LLMPipeline):
    def __init__(self, hf_model, nemo_model):
        self.hf_model = hf_model
        self.nemo_model = nemo_model

    def __call__(self, query):
        hf_result = self.hf_model(query, max_length=100, num_return_sequences=1)[0]['generated_text']
        nemo_response = nemo_model.generate(hf_result)
        return nemo_response
```

### 3. Membangun Agent untuk Pemrosesan Teks

```bash
# Inisialisasi Agent
agent = SimpleAgent(hf_model=hf_pipeline, nemo_model=nemo_model)
# Contoh Query
user_input = ”Jelaskan konsep artificial intelligence."
response = agent(user_input)
# Output
print(f"Agent Response: {response}")
```

### 4. Logika Algoritma

### LangChain Agent

- Komponen utama adalah agen yang mengambil input pengguna, mengirimkannya ke model Hugging Face GPT-2, lalu memproses output tersebut menggunakan model dari Nvidia NeMo Megatron.

### Hugging Face GPT-2

- Digunakan untuk menghasilkan respons awal dari input pengguna, kemudian hasilnya dikirim ke model yang lebih canggih (NeMo Megatron).

### Nvidia NeMo Megatron - 2

- Model ini mengambil output dari GPT-2 dan memberikan respons yang lebih mendalam atau relevan.

## Keterangan Langkah-langkah Algoritma

## Inisialisasi Hugging Face Model

- Menggunakan pipeline text-generation dari transformers untuk model GPT-2.

## Inisialisasi Nvidia NeMo Megatron Model

- `nemo.collections.nlp.models.megatron_gpt_model.MegatronGPTModel` menyediakan kemampuan generasi teks dengan model NeMo Megatron.

## Penggabungan Model dalam LangChain

- Sebuah agen dibangun dengan menggabungkan dua model (Hugging Face dan NeMo). Query pengguna diproses secara berurutan melalui kedua model.

## Pemrosesan Input

- Input pengguna diproses oleh Hugging Face GPT-2 untuk menghasilkan respons awal, yang kemudian diperkaya oleh Nvidia NeMo untuk menghasilkan keluaran akhir yang lebih komprehensif.
