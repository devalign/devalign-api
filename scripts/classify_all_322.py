"""Deep analysis and categorization of all 322 non-standard skills."""

import asyncio
import re
from sqlalchemy import text
from src.shared.database import AsyncSessionLocal


async def main() -> None:
    async with AsyncSessionLocal() as session:
        # 1. Fetch Lightcast skills
        res_lc = await session.execute(
            text("""
            SELECT s.skill_id, s.name, ss.standard_code
            FROM skills s
            JOIN skill_standards ss ON ss.skill_id = s.skill_id
            WHERE ss.standard_name = 'Lightcast';
        """)
        )
        lc_skills = res_lc.fetchall()
        lc_map = {s[1].lower(): s[1] for s in lc_skills}
        lc_norm_map = {re.sub(r"[^a-z0-9]", "", s[1].lower()): s[1] for s in lc_skills}
        lc_base_map = {}
        for s in lc_skills:
            base = re.sub(r"\s*\(.*?\)\s*", "", s[1]).strip().lower()
            if base:
                lc_base_map[base] = s[1]

        # 2. Fetch all non-standard skills
        res_non = await session.execute(
            text("""
            SELECT s.skill_id, s.name, s.nature,
                   (SELECT COUNT(*) FROM offer_skills os WHERE os.skill_id = s.skill_id) as offers,
                   (SELECT COUNT(*) FROM profile_skills ps WHERE ps.skill_id = s.skill_id) as profiles,
                   (SELECT COUNT(*) FROM cluster_skills cs WHERE cs.skill_id = s.skill_id) as clusters,
                   (SELECT COUNT(*) FROM diagnostic_skills ds WHERE ds.skill_id = s.skill_id) as diagnostics
            FROM skills s
            LEFT JOIN skill_standards ss ON ss.skill_id = s.skill_id
            WHERE ss.skill_id IS NULL
            ORDER BY offers DESC, s.name ASC;
        """)
        )
        non_skills = res_non.fetchall()

        print(f"Total non-standard skills: {len(non_skills)}")

        # Known manual mapping dictionary for Spanish, abbreviations, and common synonyms
        SYNONYM_MAP = {
            "documentation": "Software Documentation",
            "agile": "Agile Methodology",
            "softwaredecódigoabierto": "Open Source Development",
            "herramientas para la gestión de la configuración del software": "Software Configuration Management",
            "basesdedatosrelacionales": "Relational Database Management Systems",
            "puertasdeenlaceapi": "Api Gateway",
            "pruebasdecargayrendimiento": "Load Testing",
            "productmanagement": "Software Product Management",
            "productdevelopment": "Agile Product Development",
            "cloudformation": "AWS CloudFormation",
            "microsoftsql": "Microsoft SQL Servers",
            "sharepoint": "SharePoint Development",
            "jwt": "JSON Web Token (JWT)",
            "Airflow": "Apache Airflow",
            "jmeter": "Apache JMeter",
            "Kafka": "Apache Kafka",
            "TDD": "Test-Driven Development (TDD)",
            "OWASP": "Open Web Application Security Project (OWASP)",
            "Lambda": "AWS Lambda",
            "vpc": "Amazon Virtual Private Cloud (VPC)",
            "xray": "AWS X-Ray",
            "eventdriven": "Event-Driven Programming",
            "prototyping": "Software Prototyping",
            "linq": "LINQ To SQL",
            "weblogic": "Oracle WebLogic Server",
            "rails": "Ruby On Rails",
            "ruby/rails": "Ruby On Rails",
            "Jira Software": "jira",
            "Maven": "apache maven",
            "Postman": "Postman API Platform",
            "SQL Server": "Microsoft SQL Servers",
            "Stripe": "Stripe.net",
            "Mongo": "mongodb",
            "net": ".NET Development",
            "netcore": "ASP.NET Core",
            "aspnetc": "ASP.NET Core",
            "c#/net": ".NET Development",
            "ios": "Apple IOS",
            "apple": "Apple Developer Tools",
            "Windows Server": "Microsoft Windows Server Administration",
            "Windows": "Microsoft Windows Server Administration",
            "qlik": "Qlik Sense (Data Analytics Software)",
            "routing": "Network Routing",
            "switching": "Cisco Certified Network Associate (CCNA) Routing And Switching",
            "ciberseguridad": "Cybersecurity Compliance",
            "controldecalidad": "Software Quality Assurance",
            "modeladodebasesdedatos": "Relational Database Design",
            "mobiledevelopment": "Mobile Development",
            "datavisualization": "Data Visualization",
            "engineeringmanagement": "Software Engineering Management",
            "infrastructure": "Cloud Infrastructure",
            "azurekeyvault": "Azure Key Vault",
            "azureai": "Microsoft Certified: Azure AI Fundamentals",
            "azureaifoundry": "Microsoft Certified: Azure AI Fundamentals",
            "stepfunctions": "AWS Step Functions",
            "apirest": "API Design",
            "apis": "Java APIs",
            "api": "API Design",
            "CSS3": "Cascading Style Sheets (CSS)",
            "css": "Cascading Style Sheets (CSS)",
            "HTML5": "HTML",
            "html": "HTML",
            "vue": "Vue Components",
            "JBoss": "JBoss Developer Studio",
            "sftp": "Secure Shell (SSH)",
            "soap": "Simple Object Access Protocol (SOAP)",
            "cypress": "Cypress (Software)",
            "word": "Microsoft Word",
            "Word": "Microsoft Word",
            "office": "Microsoft Office",
            "Office": "Microsoft Office",
            "crm": "Customer Relationship Management (CRM) Software",
            "data analysis": "Data Analysis Software",
            "research": "Research And Development",
            "testing": "Software Testing",
        }

        mappable = []
        internal_dups = []
        junk = []
        legit_keep = []

        for row in non_skills:
            s_id, name, nature, offers, profiles, clusters, diagnostics = row
            clean = name.strip()
            lower = clean.lower()
            norm = re.sub(r"[^a-z0-9]", "", lower)

            # Check junk
            if any(
                p in lower
                for p in [
                    "noespecificado",
                    "no mencionado",
                    "not mentioned",
                    "lenguajedeprogramación",
                    "estrategiasdepruebasdealtonivel",
                    "herramientas ofimáticas",
                ]
            ):
                junk.append((name, offers, profiles, clusters))
                continue

            # Check direct synonym map
            if clean in SYNONYM_MAP or lower in SYNONYM_MAP:
                target = SYNONYM_MAP.get(clean) or SYNONYM_MAP.get(lower)
                mappable.append((name, target, "Synonym Map", offers, profiles, clusters))
                continue

            # Check Lightcast base map
            if lower in lc_base_map:
                mappable.append(
                    (name, lc_base_map[lower], "Base Name Match", offers, profiles, clusters)
                )
                continue

            # Check Lightcast norm map
            if norm in lc_norm_map and len(norm) >= 4:
                mappable.append(
                    (name, lc_norm_map[norm], "Normalized Match", offers, profiles, clusters)
                )
                continue

            # Internal duplications (e.g. Excel vs Microsoft Excel, gsuite vs Google Workspace)
            if lower in ("excel", "microsoftexcel", "excelvba"):
                internal_dups.append(
                    (name, "Consolidar en 'Microsoft Excel'", offers, profiles, clusters)
                )
                continue
            if lower in ("gsuite", "google workspace"):
                internal_dups.append(
                    (name, "Consolidar en 'Google Workspace'", offers, profiles, clusters)
                )
                continue
            if lower in ("qa",):
                internal_dups.append(
                    (
                        name,
                        "Estandarizar como 'Quality Assurance (QA)'",
                        offers,
                        profiles,
                        clusters,
                    )
                )
                continue

            # Remaining skills to keep
            legit_keep.append((name, offers, profiles, clusters))

        print(f"\n1. MAPPABLE TO LIGHTCAST: {len(mappable)}")
        for m in mappable[:25]:
            print(f"   '{m[0]}' -> '{m[1]}' ({m[2]} | offers={m[3]}, clust={m[5]})")
        print(f"   ... ({len(mappable) - 25} more)")

        print(f"\n2. INTERNAL CONSOLIDATION / STANDARDIZATION: {len(internal_dups)}")
        for d in internal_dups:
            print(f"   '{d[0]}' -> {d[1]} (offers={d[2]}, clust={d[4]})")

        print(f"\n3. JUNK / HALLUCINATIONS TO PURGE: {len(junk)}")
        for j in junk:
            print(f"   '{j[0]}' (offers={j[1]}, clust={j[3]})")

        print(f"\n4. LEGITIMATE UNIQUE SKILLS TO PRESERVE: {len(legit_keep)}")
        for k in legit_keep[:30]:
            print(f"   '{k[0]}' (offers={k[1]}, clust={k[3]})")
        print(f"   ... ({len(legit_keep) - 30} more)")


if __name__ == "__main__":
    asyncio.run(main())
