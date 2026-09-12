"""Updates all 47 clusters in Supabase with polished, distinct, professional Spanish titles."""

import asyncio
import logging
from uuid import UUID

from sqlalchemy import select, update
from src.ml_engine.infrastructure.models import ClusterModel
from src.shared.database import AsyncSessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RenameClusters")

CLUSTER_TITLES = {
    UUID("d8dc2383-279f-4115-b5f8-940655972dc1"): {
        "name": "Desarrollador Backend Java & Spring Boot",
        "description": "Especialidad en desarrollo de servicios backend empresariales y microservicios con Java, Spring Boot y SQL."
    },
    UUID("f973c87a-90cc-45e0-9c1a-74a77d546394"): {
        "name": "Desarrollador Backend Python & FastAPI",
        "description": "Especialidad en construcción de APIs de alto rendimiento, microservicios y persistencia con Python, FastAPI y PostgreSQL."
    },
    UUID("802b02e0-558f-43bf-bfe6-c50eaf34597f"): {
        "name": "Ingeniero de Software & Base de Datos SQL",
        "description": "Especialidad en ingeniería de software, procesamiento batch, servidores Linux y bases de datos relacionales SQL."
    },
    UUID("b0a8caad-a058-49ce-ab34-9ebf84cb1fd9"): {
        "name": "Científico de Datos & Machine Learning",
        "description": "Especialidad en modelado predictivo, ciencia de datos, inteligencia artificial y analítica avanzada con Python y SQL."
    },
    UUID("7e545b9a-232e-4da4-8df2-6b8d62436725"): {
        "name": "Ingeniero Cloud DevOps & Terraform",
        "description": "Especialidad en infraestructura como código (IaC), automatización de despliegues CI/CD y nubes públicas con Terraform y Kubernetes."
    },
    UUID("76dfa08b-e9b2-427e-917c-0a81877b9bb2"): {
        "name": "Desarrollador de Bases de Datos & SQL",
        "description": "Especialidad en diseño, optimización, modelado y administración de bases de datos relacionales MySQL, PostgreSQL y SQL Server."
    },
    UUID("e604db37-db25-4724-a6a8-58190eaae545"): {
        "name": "Desarrollador Full Stack React & Next.js",
        "description": "Especialidad en aplicaciones web modernas escalables con arquitectura full-stack basada en React, Next.js, TypeScript y Node.js."
    },
    UUID("27b7ef50-8375-4597-a67a-d72af1f0cb53"): {
        "name": "Desarrollador Full Stack PHP & Laravel",
        "description": "Especialidad en desarrollo de sistemas web dinámicos, paneles administrativos y APIs con Laravel, PHP y MySQL."
    },
    UUID("89f33482-8d8c-4a8e-b184-8b60a168f297"): {
        "name": "Ingeniero QA Automation (Selenium & Cypress)",
        "description": "Especialidad en diseño y ejecución de pruebas automatizadas end-to-end, pruebas funcionales y pipelines de testing con Selenium y Cypress."
    },
    UUID("c6bc693a-bf37-459b-87b7-5ba7bc686df7"): {
        "name": "Desarrollador Frontend Web & .NET",
        "description": "Especialidad en desarrollo de interfaces web enriquecidas y su integración con sistemas empresariales basados en el ecosistema .NET."
    },
    UUID("52c630c7-17cb-4e87-aa7e-b92607b45794"): {
        "name": "Analista Técnico de Sistemas & SQL",
        "description": "Especialidad en gestión de información operativa, soporte técnico empresarial y consultas estructuradas en bases de datos relacionales."
    },
    UUID("8c6738e3-8088-4110-a25a-6227d789fb0c"): {
        "name": "Analista de Datos & Inteligencia de Negocios",
        "description": "Especialidad en modelado analítico, extracción de insights de negocio y construcción de tableros de control con Power BI y SQL."
    },
    UUID("06c47170-320c-4322-b808-71f5dcb1d848"): {
        "name": "Desarrollador Mobile Flutter & Full Stack",
        "description": "Especialidad en creación de aplicaciones móviles multiplataforma de alto impacto con Flutter integradas a backends web."
    },
    UUID("8ac5f932-1fda-45c5-b82f-e84bf6c5b890"): {
        "name": "Ingeniero Cloud Azure & DevOps",
        "description": "Especialidad en administración de servicios en la nube Microsoft Azure, automatización de integración continua y despliegue continuo."
    },
    UUID("ebc61d45-4ba5-46b3-9793-08895e1a5cfc"): {
        "name": "Desarrollador Frontend React & Vue.js",
        "description": "Especialidad en desarrollo de aplicaciones de página única (SPA) con los principales frameworks modernos de JavaScript."
    },
    UUID("21923bff-d9ac-4f34-a594-5cadcd5dab14"): {
        "name": "Arquitecto Cloud Multi-Nube (AWS, Azure, GCP)",
        "description": "Especialidad en diseño y gobernanza de arquitecturas distribuidas en entornos multi-cloud con AWS, Azure y Google Cloud Platform."
    },
    UUID("320f93b9-2a7c-4ca8-a4d6-410186ab2027"): {
        "name": "Diseñador UI/UX & Maquetador Web",
        "description": "Especialidad en diseño centrado en el usuario, prototipado interactivo en Figma y maquetación web responsiva con HTML5 y CSS3."
    },
    UUID("201a0af7-2a00-4eef-936e-68c12c892d50"): {
        "name": "Ingeniero de Plataforma & Contenedores",
        "description": "Especialidad en orquestación de contenedores, observabilidad de infraestructura y confiabilidad operativa con Docker y Kubernetes."
    },
    UUID("23950255-96ce-4f53-a56e-abcf17f3dba5"): {
        "name": "Científico de Datos Junior & Power BI",
        "description": "Especialidad en análisis exploratorio de datos, limpieza de datasets y visualización ejecutiva combinando Python, SQL y Power BI."
    },
    UUID("dceaa17e-78c1-44c2-a420-394ec4e42e02"): {
        "name": "Analista de Aseguramiento de Calidad (QA)",
        "description": "Especialidad en control de calidad integral de software, diseño de planes de prueba, prevención de defectos y criterios de aceptación."
    },
    UUID("b50348a4-5672-4b7e-acd2-9d11fdc7176a"): {
        "name": "Analista QA Funcional & Metodologías Ágiles",
        "description": "Especialidad en validación funcional, gestión de historias de usuario y seguimiento de incidentes en entornos ágiles con Jira."
    },
    UUID("f948e30f-d562-481b-9d89-e3b4955826ed"): {
        "name": "Especialista BI Empresarial (Tableau & Power BI)",
        "description": "Especialidad en analítica visual corporativa, reportería estratégica y gobernanza de métricas con Tableau, Power BI y SQL."
    },
    UUID("24b39abd-ac1e-45f2-aaa6-8ba9da3223a7"): {
        "name": "Desarrollador Full Stack Node.js & React",
        "description": "Especialidad en desarrollo integral de soluciones web combinando servicios desacoplados en Node.js con interfaces interactivas en React."
    },
    UUID("49948a2d-89c6-444f-91e9-1cbda8158b2e"): {
        "name": "Desarrollador de Software Multi-Lenguaje",
        "description": "Especialidad en programación políglota, soporte de aplicaciones legadas y automatización de procesos con Java, Python y SQL."
    },
    UUID("3d33957c-0c76-49c4-8a0c-40737df4ccc6"): {
        "name": "Desarrollador Web Frontend (JavaScript & TypeScript)",
        "description": "Especialidad en construcción de interfaces de usuario modernas, accesibilidad web y tipado estricto con TypeScript y JavaScript."
    },
    UUID("07ce8091-32e9-47c0-847e-d3f3bca2854e"): {
        "name": "Analista de Reportes y Dashboards Power BI",
        "description": "Especialidad en modelado DAX, diseño de tableros de control operativo e integración de fuentes de datos directas con Power BI."
    },
    UUID("7d8d8687-2ed3-400b-8146-b876b58bfe66"): {
        "name": "Desarrollador Backend Node.js & TypeScript",
        "description": "Especialidad en arquitectura de servicios backend, consumo de microservicios y bases de datos relacionales con TypeScript y Node.js."
    },
    UUID("54241e00-ac31-40e4-879a-724897e1da84"): {
        "name": "Ingeniero Backend & Microservicios (Golang / Java)",
        "description": "Especialidad en sistemas distribuidos de baja latencia, caching distribuido con Redis y despliegue contenerizado en Kubernetes."
    },
    UUID("2610a0d3-790a-4eb0-bdb4-c1eb2745eb92"): {
        "name": "Desarrollador Frontend React & Mobile",
        "description": "Especialidad en interfaces web responsivas de alta fidelidad y desarrollo de aplicaciones móviles híbridas con el ecosistema React."
    },
    UUID("6a2d3807-e76c-41b6-8837-0aa73711d8cc"): {
        "name": "Desarrollador Móvil iOS & Android (Kotlin / Swift)",
        "description": "Especialidad en ingeniería de aplicaciones móviles nativas para las plataformas iOS y Android con Swift y Kotlin."
    },
    UUID("e4a35fc9-7a7e-43bf-95e7-7eb6faef644d"): {
        "name": "Ingeniero QA de Integración Continua (CI & Testing)",
        "description": "Especialidad en integración de baterías de pruebas automáticas en pipelines de Jenkins, control de versiones y calidad continua."
    },
    UUID("bae8167e-21cb-468f-874c-3f9961d1a5ab"): {
        "name": "Desarrollador Full Stack .NET & Angular",
        "description": "Especialidad en plataformas empresariales robustas integrando backends en C# / .NET Core con clientes web avanzados en Angular."
    },
    UUID("37f804ca-008c-4c04-b79b-48e328dbe31d"): {
        "name": "Desarrollador Full Stack Ruby on Rails",
        "description": "Especialidad en prototipado rápido y desarrollo ágil de productos web escalables con Ruby on Rails y componentes React."
    },
    UUID("82ca4df6-e985-4b4c-baea-16bd0bc18e09"): {
        "name": "Desarrollador Mobile Nativo Android (Kotlin & Java)",
        "description": "Especialidad en arquitectura de software para dispositivos Android, librerías Jetpack y rendimiento con Kotlin y Java."
    },
    UUID("5167f663-0265-4b75-bc5a-0f24f35f32cc"): {
        "name": "Ingeniero DevOps & Gestión Ágil (SRE)",
        "description": "Especialidad en confiabilidad del servicio (SRE), automatización de infraestructura y optimización del flujo de entrega con Kanban y CI/CD."
    },
    UUID("75b84191-6a70-4391-a075-840289c630e3"): {
        "name": "Desarrollador Frontend Angular & TypeScript",
        "description": "Especialidad en desarrollo modular y arquitectura empresarial basada en Angular, RxJS y TypeScript para el sector corporativo."
    },
    UUID("600fc3f5-5695-4c44-ad79-b43fc06edd40"): {
        "name": "Administrador de Sistemas Unix / Linux",
        "description": "Especialidad en configuración, seguridad, scripting de shell y mantenimiento de servidores en entornos de producción Unix y Linux."
    },
    UUID("03176481-7420-43aa-876b-3e5def784680"): {
        "name": "Desarrollador Cloud AWS & Python",
        "description": "Especialidad en arquitecturas serverless y desarrollo de servicios en la nube con Amazon Web Services, Python y bases de datos SQL."
    },
    UUID("97f9d9f6-f7e3-4e0e-a6d8-ba93a905e717"): {
        "name": "Analista de Procesos de Software & GeneXus",
        "description": "Especialidad en desarrollo de sistemas administrativos mediante plataformas low-code GeneXus y gestión ágil de flujos de trabajo."
    },
    UUID("8d85cb0a-9509-478c-8379-3b9985111d65"): {
        "name": "Desarrollador Python & Soluciones de IA",
        "description": "Especialidad en automatización inteligente de flujos de trabajo, integración de modelos de lenguaje e ingeniería con Python y SQL."
    },
    UUID("e08d7bac-12d6-48e2-8e93-76a7ea1c670f"): {
        "name": "Desarrollador Backend Java Microservicios",
        "description": "Especialidad en desarrollo de componentes backend desacoplados, contratos API y gestión de dependencias con Spring Boot y Jira."
    },
    UUID("a840c873-ed45-409b-a1f6-1e4eac266713"): {
        "name": "Administrador de Infraestructura Cloud Azure & M365",
        "description": "Especialidad en gestión de identidad, servidores virtuales, servicios de directorio activo y soporte de nivel 3 en el ecosistema Microsoft."
    },
    UUID("d244145e-a716-4cd4-bd06-8022e10441d8"): {
        "name": "Analista Programador de Aplicaciones Web",
        "description": "Especialidad en desarrollo e integración de módulos empresariales, mantenimiento de sistemas y bases de datos relacionales."
    },
    UUID("ad932fc4-d8f5-4c00-b61c-e5c47db75b7e"): {
        "name": "Desarrollador Web WordPress & PHP",
        "description": "Especialidad en creación de sitios web dinámicos, comercio electrónico con WooCommerce y personalización de temas y plugins en PHP."
    },
    UUID("2e9e0f01-55d8-4eab-b250-ffa75ee4aeac"): {
        "name": "Administrador de Base de Datos PostgreSQL & Linux",
        "description": "Especialidad en afinamiento de consultas, alta disponibilidad, respaldos y administración de motores PostgreSQL sobre entornos Linux."
    },
    UUID("38076a46-aa79-4524-96a8-2b95911e579d"): {
        "name": "Líder Técnico de Producto & Metodologías Ágiles",
        "description": "Especialidad en alineación tecnológica de producto, priorización de backlog en Jira y coordinación de equipos ágiles multidisciplinarios."
    },
    UUID("e236c929-e2be-49d1-87ae-d59fd902c3a2"): {
        "name": "Consultor Técnico & Aseguramiento de Calidad",
        "description": "Especialidad en consultoría de software, adopción de marcos ágiles y estándares de calidad técnica en proyectos de transformación digital."
    }
}


async def main():
    logger.info("Applying clean, professional Spanish titles to all 47 clusters in Supabase...")
    async with AsyncSessionLocal() as session:
        query = select(ClusterModel)
        res = await session.execute(query)
        clusters = res.scalars().all()
        
        updated_count = 0
        for c in clusters:
            info = CLUSTER_TITLES.get(c.cluster_id)
            if info:
                await session.execute(
                    update(ClusterModel)
                    .where(ClusterModel.cluster_id == c.cluster_id)
                    .values(
                        name=info["name"],
                        description=info["description"]
                    )
                )
                logger.info(f"Updated: {info['name']}")
                updated_count += 1
            else:
                logger.warning(f"Cluster {c.cluster_id} not in map!")

        await session.commit()
        logger.info(f"Successfully updated {updated_count} / {len(clusters)} clusters in Supabase!")


if __name__ == "__main__":
    asyncio.run(main())
