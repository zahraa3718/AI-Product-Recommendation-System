# 1. تثبيت المكتبات المطلوبة
!pip install gradio pandas scikit-learn plotly -q

import pandas as pd
import numpy as np
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# 2. إنشاء قاعدة بيانات منتجات (مثال: متجر إلكترونيات)
products_data = {
    'Product_ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Product_Name': [
        'Apple MacBook Pro 16"',
        'Dell XPS 15 Laptop',
        'Apple iPad Pro 12.9"',
        'Sony WH-1000XM5 Headphones',
        'Apple AirPods Max',
        'Samsung Galaxy S23 Ultra',
        'iPhone 15 Pro Max',
        'Logitech MX Master 3S Mouse',
        'Keychron K2 Mechanical Keyboard',
        'Asus ROG Gaming Laptop'
    ],
    'Category': [
        'Laptops', 'Laptops', 'Tablets', 'Audio', 'Audio', 
        'Smartphones', 'Smartphones', 'Accessories', 'Accessories', 'Laptops'
    ],
    'Description': [
        'Powerful M3 Max chip laptop for professional video editing and developers.',
        'High performance Intel i9 laptop with 4K OLED screen for creators.',
        'M2 tablet with Liquid Retina display perfect for drawing and digital art.',
        'Premium wireless noise cancelling headphones with long battery life.',
        'Over-ear wireless headphones with high-fidelity audio and active noise cancellation.',
        'Flagship Android smartphone with S-Pen, 200MP camera, and fast processor.',
        'Premium Apple smartphone with Titanium design, A17 Pro chip, and advanced camera.',
        'Ergonomic wireless mouse with silent clicks and high precision sensor.',
        'Wireless mechanical keyboard with tactile switches and RGB backlighting.',
        'High-end gaming laptop with RTX 4090 GPU and ultra fast refresh rate screen.'
    ]
}

df = pd.DataFrame(products_data)

# 3. معالجة النصوص وحساب التشابه بواسطة TF-IDF و Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
df['Combined_Features'] = df['Category'] + " " + df['Description']
tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])

# حساب مصفوفة التشابه بين كل المنتجات
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. دالة التوصية وتوليد النتائج والرسم البياني
def recommend_products(selected_product_name):
    idx = df[df['Product_Name'] == selected_product_name].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:5]  # أخذ أفضل 4 منتجات تشابهاً
    
    product_indices = [i[0] for i in sim_scores]
    similarity_values = [round(i[1] * 100, 1) for i in sim_scores]
    
    recommended_df = df.iloc[product_indices][['Product_Name', 'Category', 'Description']].copy()
    recommended_df['نسبة التوافق'] = [f"{score}%" for score in similarity_values]
    
    # رسم بياني لنسب التشابه
    fig = px.bar(
        x=similarity_values,
        y=recommended_df['Product_Name'],
        orientation='h',
        labels={'x': 'نسبة التوافق %', 'y': 'المنتج المقترح'},
        title=f"🎯 أفضل المنتجات المقترحة لـ: {selected_product_name}",
        color=similarity_values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return recommended_df, fig

# 5. بناء الواجهة التفاعلية بواسطة Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 نظام التوصية بالمنتجات الذكي (AI Product Recommender)")
    gr.Markdown("اختر أي منتج لاختبار خوارزمية التوصية ورؤية النتائج المقترحة ورسم التوافق:")
    
    with gr.Row():
        with gr.Column():
            product_dropdown = gr.Dropdown(
                choices=list(df['Product_Name']), 
                value=df['Product_Name'][0], 
                label="اختر منتجاً (Select Product)"
            )
            btn = gr.Button("🔮 إظهار التوصيات الذكية", variant="primary")
            
        with gr.Column():
            output_table = gr.Dataframe(label="قائمة التوصيات المقترحة")
            output_plot = gr.Plot(label="تحليل نسبة التوافق والتشابه")
            
    btn.click(recommend_products, inputs=[product_dropdown], outputs=[output_table, output_plot])

# تشغيل التطبيق
demo.launch(share=True)
