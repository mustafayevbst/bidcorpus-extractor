from pydantic import BaseModel, Field

class Tender(BaseModel):
    customer_name: str | None = Field(
        None, description = "Название организации-заказчика"
    )
    tender_number: str | None = Field(
        None, description = "Номер тендера, например '002/2013'"
    )
    category: str | None = Field(
        None, description="Категория закупки кратко, например 'fuel', 'office supplies'"
    )
    deadline: str | None = Field(
        None, description = "Дата вскрытия конвертов в формате YYYY-MM-DD"
    )
    requirements: list[str] = Field(
        default_factory=list,
        description="Список требований к участникам (документы, лицензий)",
    )
    participation_conditions: list[str] = Field(
        default_factory = list,
        description = "Условия участия в тендере",
    )