from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils import timezone


# === CUSTOM USER MODEL (Company as Client) ===
class CompanyManager(BaseUserManager):
    def create_user(self, email, company_name, password=None, **extra_fields):
        if not email:
            raise ValueError("Email is required")
        if not company_name:
            raise ValueError("Company name is required")
        email = self.normalize_email(email)
        user = self.model(email=email, company_name=company_name, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, company_name, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, company_name, password, **extra_fields)


class CompanyUser(AbstractBaseUser, PermissionsMixin):
    company_name = models.CharField(max_length=255, unique=True)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["company_name"]

    objects = CompanyManager()

    def __str__(self):
        return self.company_name


# === CRM MODELS ===

class Customer(models.Model):
    owner = models.ForeignKey(CompanyUser, on_delete=models.CASCADE, related_name="customers")
    name = models.CharField(max_length=255)
    email = models.EmailField()
    company = models.CharField(max_length=255, blank=True, null=True)
    industry = models.CharField(max_length=255, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    product_interested = models.CharField(max_length=255, blank=True, null=True)
    revenue_potential = models.FloatField(default=0)
    conversion_rate = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.owner.company_name})"


class CustomerReview(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="reviews")
    owner = models.ForeignKey(CompanyUser, on_delete=models.CASCADE, related_name="reviews")
    review_text = models.TextField()
    sentiment_score = models.FloatField(default=0.0)
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review for {self.customer.name}"


class ModelFeedback(models.Model):
    owner = models.ForeignKey(CompanyUser, on_delete=models.CASCADE, related_name="feedbacks")
    model_name = models.CharField(max_length=50)
    feedback_text = models.TextField()
    rating = models.IntegerField(default=0)
    date_added = models.DateTimeField(auto_now_add=True)
