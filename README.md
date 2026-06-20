# Devalign API

> API con capacidades de Machine Learning para el análisis de brechas de habilidades técnicas y alineación profesional de desarrolladores en Perú.

## Overview

Devalign analiza la demanda del mercado laboral (vía scraping offline) y los perfiles individuales de los desarrolladores (a través del análisis de CV) para:

1. Identificar la especialidad técnica del desarrollador mediante agrupamiento en clústeres (UMAP + HDBSCAN).
2. Detectar brechas de habilidades comparándolas con el estándar de mercado para su especialidad usando la métrica Weighted Jaccard.
3. Generar recomendaciones y planes de acción priorizados determinísticamente ($Prioridad = Peso \times Frecuencia$).

## Quick Start

```bash
# Requiere: Python 3.12+ y uv
# Instalar uv si no está presente
pip install uv

# Instalar dependencias del proyecto
make install

# Configurar variables de entorno
cp .env.example .env
# Edite .env con sus credenciales de Supabase, OpenAI, y Voyage AI

# Iniciar servidor de desarrollo
make dev
# → http://localhost:8000/api/v1/docs
```

## API Endpoints

| Método | Endpoint | Descripción |
|:---|:---|:---|
| GET | `/health` | Verificación de estado de la API |
| GET | `/api/v1/users/me` | Perfil del usuario autenticado (JIT Provisioning) |
| POST | `/api/v1/users/me/cv` | Carga de CV en PDF/DOCX e inicio de procesamiento |
| GET | `/api/v1/profile/me` | Obtención del JSON de Diagnóstico y brechas |
| PUT | `/api/v1/profile/skills` | Actualización manual de habilidades del perfil |
| GET | `/api/v1/profile/skills-graph` | Grafo de conocimiento de habilidades del usuario |

## Arquitectura

```
src/
├── delivery/       # Rutas HTTP, autenticación JWT, y controladores de entrada
├── ml_engine/      # Extracción estructurada con LLM, embeddings (Voyage AI), normalización y alineación (Weighted Jaccard)
├── scraper/        # Adquisición de ofertas de empleo (procesamiento offline)
└── shared/         # Base de datos (SQLAlchemy/Alembic), configuración y utilidades
```

Diseño de arquitectura limpia con flujo de control: `domain → application → infrastructure → interface`.

## Tech Stack

- **FastAPI** + Pydantic v2 + pydantic-settings
- **Supabase** (PostgreSQL + pgvector + Auth + Storage)
- **Voyage AI** (Embeddings de 1024 dimensiones)
- **OpenAI API / Claude API** (Extracción estructurada JSON)
- **scikit-learn** + **umap-learn** + **hdbscan** (Clustering offline de vacantes)
- **pypdf** + **python-docx** (Extracción de texto de CVs)
- **SQLAlchemy 2.0** + **asyncpg** + **pgvector-python** (Persistencia y búsqueda vectorial asíncrona)
- **Alembic** (Single Source of Truth para el esquema relacional en Postgres)
- **uv** + Ruff + pytest

## Documentación del Proyecto

La documentación de diseño, decisiones arquitectónicas, base de datos y contratos está centralizada en el repositorio de documentación central:

- [🏗️ Arquitectura Técnica](../devalign-docs/ARCHITECTURE.md)
- [🤝 Contratos de Interfaz](../devalign-docs/CONTRACTS.md)
- [🗄️ Modelo de Base de Datos](../devalign-docs/DATABASE.md)
- [🧠 Lógica Core e Inferencia](../devalign-docs/MODEL.md)
- [🗺️ Roadmap de Producto](../devalign-docs/ROADMAP.md)
- [🎯 Alcance MVP](../devalign-docs/SCOPE.md)
