# Informe de Proyecto

## 1. Título Propuesto

**Aplicación web con motor de inferencia basado en Machine Learning para el analisis de competencias tecnicas demandadas por el sector IT orientado a desarrolladores del Perú**

---

## 2. Problemática y Alcance

### Problema
Existe una asimetría de información crítica entre la formación técnica de los desarrolladores y la demanda real del sector IT. El problema no es solo la falta de experiencia, sino la **desalineación técnica**: los desarrolladores enfrentan una sobrecarga de información y carecen de una ruta clara para especializarse en los nichos tecnológicos (stacks) que el mercado realmente demanda y valora.

### Propósito
Desarrollar un sistema inteligente que permita identificar automáticamente la especialidad técnica de un desarrollador, diagnosticar su nivel de alineación con el mercado actual y generar una ruta de especialización profunda basada en datos reales de la industria.

### Alcance
El sistema será capaz de:

- **Inteligencia de Mercado:** Recolectar y procesar miles de ofertas laborales mediante Web Scraping para descubrir clústeres tecnológicos vivos.
- **Profiling Invisible:** Procesar el CV del usuario mediante NLP para identificar su afinidad técnica sin necesidad de formularios extensos.
- **Detección de Brechas (Multi-Afinidad):** Identificar las habilidades faltantes (hard y soft) comparando el perfil del usuario contra el "Universo de Competencias" de su especialidad.
- **Roadmap de Especialización:** Generar una ruta de aprendizaje progresiva y lógica mediante IA Generativa (GenAI), estructurando el pool de brechas en niveles de madurez técnica.

---

## 3. Arquitectura de Componentes (Enfoque Software)

El sistema se implementará bajo una arquitectura modular desacoplada basada en microservicios utilizando **FastAPI**.

### 3.1 Componente de Adquisición (Scraper)
Responsable de la extracción masiva de datos de portales de empleo de alta relevancia (LinkedIn, GetOnBoard, Computrabajo).

**Especificaciones técnicas:**
- **Volumen de datos:** Procesamiento de una muestra representativa de 5,000 ofertas de empleo activas en el sector TI.
- **Patrón de diseño:** Implementación de *Strategy Pattern* para garantizar la escalabilidad y adaptabilidad ante diversos esquemas de DOM.
- **Extracción de entidades:** Identificación de `hard_skills`, `soft_skills`, herramientas de software y frameworks mediante técnicas de normalización léxica.

---

### 3.2 Componente de Inteligencia (Motor Analítico ML)

#### a) Descubrimiento de Especialidades (Clustering)
Utilización del algoritmo **K-Prototypes** para el agrupamiento de variables mixtas (numéricas y categóricas). El modelo trasciende la clasificación convencional para descubrir "Especialidades Técnicas Empíricas" (ej. *Backend Cloud-Native*, *Frontend Modern UX*), determinando la co-ocurrencia de tecnologías en el mercado real.

#### b) Vectorización y Profiling Semántico
Procesamiento del currículum vítae (CV) mediante modelos de *Embeddings*. El documento se vectoriza para calcular la **afinidad semántica** (Similitud de Coseno) respecto a los centroides de los clústeres identificados, permitiendo una detección precisa del perfil actual y potencial del desarrollador.

#### c) Motor de Diagnóstico de Brechas
Análisis multidimensional que contrasta el vector del usuario contra el "Universo de Competencias" de su especialidad objetivo, identificando discrepancias técnicas (gaps) a nivel de herramientas, lenguajes y metodologías.

---

### 3.3 Componente de Entrega (API REST)
Expone los resultados al frontend, gestionando el flujo de diagnóstico y generación de roadmaps on-demand.

---

### 3.4 Arquitectura de IA Generativa con RAG y LangChain
La generación del *Roadmap de Aprendizaje Personalizado* se fundamenta en una arquitectura de **Generación Aumentada por Recuperación (RAG)**, orquestada mediante el framework **LangChain**. Este enfoque garantiza que la salida del modelo no sea genérica, sino técnica, estructurada y estandarizada.

**Componentes del flujo RAG con LangChain:**
- **Indexación y Almacenamiento:** Los estándares globales **SFIA 9** (*Skills Framework for the Information Age*) y **IEEE SWECOM** (*Software Engineering Competency Model*) son indexados y almacenados en una base de datos vectorial.
- **Recuperación Contextual (Retrieve):** El sistema recupera el contexto exacto de estos estándares basándose en la especialidad técnica del usuario y el nivel de *seniority* detectado.
- **Inyección y Generación:** LangChain inyecta este contexto técnico en el prompt del LLM (*Large Language Model*), asegurando que la ruta de aprendizaje generada esté estrictamente alineada con los estándares internacionales y carezca de alucinaciones.

**Flujo de Orquestación (LangChain Pipeline):**
```python
# Ejemplo conceptual del flujo de orquestación
chain = (
    {
        "context": retriever | format_docs,
        "specialty": RunnablePassthrough(),
        "seniority": RunnablePassthrough(),
        "gaps": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)
```

### 3.5 Validación Taxonómica del Modelo ML
El marco teórico valida el modelo de Machine Learning mediante un proceso de contraste taxonómico:
- Los clústeres de tecnologías empíricas obtenidos mediante el scraping se validan utilizando las taxonomías oficiales de **SFIA 9** (ej. la habilidad *PROG: Programming/Software Development*) y las áreas de conocimiento de **SWEBOK/SWECOM**.
- Este proceso asegura que las habilidades extraídas del mercado real se correspondan con competencias validadas tanto por la industria como por la academia global.

### 3.6 Control de Seniority en la Ruta de Aprendizaje
El componente de GenAI utiliza el RAG para segmentar la complejidad y profundidad del roadmap generado:
- **Niveles Junior / Initial:** Se asocian a los niveles 1-3 de responsabilidad de SFIA 9 y niveles iniciales de SWECOM, priorizando fundamentos de construcción de software y lógica algorítmica.
- **Niveles Senior / Advanced:** Se asocian a los niveles 4-5 de SFIA 9 y niveles avanzados de SWECOM, priorizando arquitectura de sistemas, diseño de soluciones complejas y gestión estratégica.

### 3.7 Componente de Gestión de Identidad y Persistencia (Supabase)
Se implementa una infraestructura de servicios basada en **Supabase** para la gestión crítica de datos y usuarios:
- **Gestión de Identidad (Auth):** Sistema de autenticación centralizado con soporte para **OAuth 2.0 via Google**, permitiendo un registro seguro y simplificado.
- **Almacenamiento de Documentos (Storage):** Repositorio de objetos para la persistencia de los currículums vítae cargados, garantizando la integridad de los datos para auditoría técnica.
- **Capa de Datos:** Base de datos relacional PostgreSQL para el almacenamiento de vectores de perfilamiento y metadatos de progreso.

---

## 4. Flujo de Funcionamiento Sistémico

1. **Ingesta de Datos:** El Scraper recolecta y normaliza 5,000 registros de la demanda laboral actual.
2. **Modelado ML:** El algoritmo K-Prototypes descubre clústeres de especialidades técnicas emergentes.
3. **Profiling de Usuario:** El CV del usuario se vectoriza para determinar su afinidad semántica con los clústeres.
4. **Análisis de Brechas:** Se contrastan las competencias del usuario con el estándar del mercado.
5. **Orquestación RAG:** LangChain recupera el contexto de SFIA 9 y SWECOM según el seniority y especialidad elegida.
6. **Generación de Roadmap:** El LLM genera una ruta de aprendizaje técnica, jerarquizada y validada taxonómicamente.
7. **Entrega de Resultados:** La API REST sirve el diagnóstico y la ruta de especialización al usuario final.

## 5. Especificación de la Respuesta del Modelo (Entregable Técnico)

```json
{
  "perfil": {
    "especialidad_detectada": "Backend Java Cloud-Native",
    "afinidad_score": 0.85,
    "afinidades_secundarias": ["DevOps", "Fullstack JS"]
  },
  "diagnostico_alineacion": {
    "competencias_consolidadas": ["Java 17", "Spring Boot", "SQL"],
    "pool_de_brechas": [
      {
        "skill": "Kubernetes",
        "tipo": "Hard Skill",
        "importancia_mercado": "Crítica"
      },
      {
        "skill": "Arquitectura de Microservicios",
        "tipo": "Metodología",
        "importancia_mercado": "Alta"
      }
    ]
  },
  "roadmap_estrategico": {
    "fase_1": "Fundamentos de Cloud y Contenedores (Docker)",
    "fase_2": "Orquestación y Despliegue (K8s / AWS)",
    "fase_3": "Patrones de Resiliencia y Observabilidad"
  }
}
```

## 6. Sustento de Innovación

### Propuesta Diferenciadora
A diferencia de plataformas de aprendizaje genéricas, este sistema utiliza **Inteligencia de Mercado** para que el aprendizaje no sea teórico, sino dictado por la demanda real y actual del Sector IT.

### Innovaciones Clave
- **Descubrimiento Dinámico de Roles:** El mercado define qué es un especialista, no un currículo estático.
- **Profiling Invisible:** Diagnóstico basado en el CV real, eliminando sesgos de autopercepción.
- **Especialización Profunda:** Enfoque en nichos de alta demanda para maximizar la empleabilidad.

## 7. Impacto Esperado

- **Para desarrolladores:** Reducción del tiempo de búsqueda de empleo y mayor claridad en el crecimiento técnico.
- **Para el Sector IT:** Reducción de la brecha de talento cualificado en tecnologías específicas.

## 8. Tecnologías Sugeridas
- **Backend:** FastAPI (Python).
- **IA/ML:** K-Prototypes, Scikit-learn, LangChain (GenAI), OpenAI Embeddings.
- **Persistencia y Auth:** Supabase (PostgreSQL, Auth via Google, Storage para CVs).
- **Infraestructura:** Docker, GitHub Actions (CI/CD).

## 9. Diseño de Interacción (UX)
El sistema prioriza la **Experiencia One-Click**:
1. El usuario carga su CV y el sistema hace el trabajo pesado de perfilamiento en segundo plano.
2. Se presenta un **Dashboard de Potencial** donde el usuario ve sus afinidades y elige su ruta de especialización, disparando la generación del roadmap personalizado de forma inmediata.

## 10. Conclusión
La implementación de este sistema representa una solución de ingeniería avanzada al problema de la desalineación técnica en el Perú. Al combinar Machine Learning para el análisis de mercado y GenAI para la mentoría técnica, se ofrece una herramienta poderosa que transforma el caos de la demanda laboral en una ruta clara, objetiva y accionable para el desarrollo profesional.