import os
import django
from datetime import datetime

# ✅ Set up Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SalesInsight.settings")  # 🔹 replace with your project name
django.setup()

from core.models import Customer, CustomerReview, ModelFeedback


def insert_sample_customers():
    customers = [
        Customer(name='John Doe', email='john.doe@example.com', company='TechNova Inc', industry='Software', country='USA', product_interested='SmartCRM Pro', revenue_potential=15000, conversion_rate=0.65),
        Customer(name='Emma Watson', email='emma.watson@example.com', company='BrightWorks', industry='Marketing', country='UK', product_interested='AI Campaign Manager', revenue_potential=12000, conversion_rate=0.72),
        Customer(name='Liam Smith', email='liam.smith@example.com', company='AeroLogic', industry='Aviation', country='Canada', product_interested='Predictive Analytics Suite', revenue_potential=18000, conversion_rate=0.68),
        Customer(name='Sophia Brown', email='sophia.brown@example.com', company='FinEdge', industry='Finance', country='India', product_interested='SmartCRM Pro', revenue_potential=22000, conversion_rate=0.70),
        Customer(name='Noah Wilson', email='noah.wilson@example.com', company='EduCloud', industry='EdTech', country='Australia', product_interested='Learning Insights AI', revenue_potential=14000, conversion_rate=0.75),
        Customer(name='Olivia Johnson', email='olivia.johnson@example.com', company='MediSync', industry='Healthcare', country='Germany', product_interested='HealthAI Tracker', revenue_potential=26000, conversion_rate=0.80),
        Customer(name='William Jones', email='william.jones@example.com', company='Retail360', industry='E-commerce', country='France', product_interested='Customer Pulse', revenue_potential=17000, conversion_rate=0.60),
        Customer(name='Ava Garcia', email='ava.garcia@example.com', company='AutoEdge', industry='Automotive', country='Japan', product_interested='Predictive Maintenance AI', revenue_potential=19500, conversion_rate=0.78),
        Customer(name='James Miller', email='james.miller@example.com', company='AgriTech', industry='Agriculture', country='Brazil', product_interested='Crop Monitor AI', revenue_potential=13000, conversion_rate=0.64),
        Customer(name='Mia Martinez', email='mia.martinez@example.com', company='BuildSmart', industry='Construction', country='Spain', product_interested='Project IQ', revenue_potential=21000, conversion_rate=0.82),
        Customer(name='Benjamin Davis', email='benjamin.davis@example.com', company='EcoFuture', industry='Energy', country='Norway', product_interested='SolarFlow Optimizer', revenue_potential=25000, conversion_rate=0.76),
        Customer(name='Charlotte Taylor', email='charlotte.taylor@example.com', company='InnovaMedia', industry='Media', country='USA', product_interested='Content Genie', revenue_potential=16000, conversion_rate=0.73),
        Customer(name='Lucas Hernandez', email='lucas.hernandez@example.com', company='TransLogix', industry='Logistics', country='Mexico', product_interested='RouteAI Optimizer', revenue_potential=17500, conversion_rate=0.66),
        Customer(name='Amelia Moore', email='amelia.moore@example.com', company='FoodChainX', industry='FoodTech', country='Italy', product_interested='SupplySense AI', revenue_potential=15500, conversion_rate=0.69),
        Customer(name='Elijah Thomas', email='elijah.thomas@example.com', company='CyberCore', industry='Cybersecurity', country='Singapore', product_interested='ThreatShield AI', revenue_potential=28000, conversion_rate=0.85),
        Customer(name='Harper Anderson', email='harper.anderson@example.com', company='StyleBot', industry='Fashion', country='South Korea', product_interested='TrendPredictor AI', revenue_potential=12500, conversion_rate=0.58),
        Customer(name='Henry Jackson', email='henry.jackson@example.com', company='SmartHomez', industry='IoT', country='Sweden', product_interested='HomeSense Hub', revenue_potential=23000, conversion_rate=0.79),
        Customer(name='Evelyn White', email='evelyn.white@example.com', company='AquaFlow', industry='Utilities', country='Netherlands', product_interested='WaterSense Pro', revenue_potential=19000, conversion_rate=0.67),
        Customer(name='Alexander Martin', email='alexander.martin@example.com', company='AdaptoTech', industry='AI Solutions', country='USA', product_interested='SmartCRM Pro', revenue_potential=27000, conversion_rate=0.83),
        Customer(name='Ella Thompson', email='ella.thompson@example.com', company='VisionAI', industry='Analytics', country='India', product_interested='Insightify Pro', revenue_potential=20000, conversion_rate=0.74),
    ]

    Customer.objects.bulk_create(customers)
    print("✅ 20 sample customers inserted successfully!")


def insert_sample_reviews():
    reviews = [
        (1, "Amazing CRM! It helped us increase lead conversion by 30%.", 0.9),
        (2, "The AI campaign manager works well but could be faster.", 0.7),
        (3, "Predictive analytics gave us great insights for our aviation projects.", 0.8),
        (4, "SmartCRM Pro is intuitive and efficient.", 0.85),
        (5, "Learning Insights AI improved student engagement significantly.", 0.88),
    ]

    for cust_id, text, score in reviews:
        try:
            customer = Customer.objects.get(pk=cust_id)
            CustomerReview.objects.create(customer=customer, review_text=text, sentiment_score=score)
        except Exception as e:
            print(f"❌ Skipped review for customer {cust_id}: {e}")

    print("✅ 5 sample reviews inserted successfully!")


def insert_model_feedback():
    feedbacks = [
        ModelFeedback(model_name='lead_analyzer', feedback_text='Predictions were accurate and helpful.', rating=5),
        ModelFeedback(model_name='lead_analyzer', feedback_text='Could be improved for smaller datasets.', rating=4),
        ModelFeedback(model_name='sales_pitch', feedback_text='Generated pitch was engaging and natural.', rating=5),
        ModelFeedback(model_name='sales_pitch', feedback_text='Sometimes repetitive language.', rating=3),
        ModelFeedback(model_name='lead_analyzer', feedback_text='Good accuracy overall, but slow response time.', rating=4),
    ]
    ModelFeedback.objects.bulk_create(feedbacks)
    print("✅ 5 model feedback entries inserted successfully!")


if __name__ == "__main__":
    insert_sample_customers()
    insert_sample_reviews()
    insert_model_feedback()
