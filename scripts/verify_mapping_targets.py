"""Verify all mapping sources and targets against skills table."""

import asyncio
from sqlalchemy import text
from src.shared.database import AsyncSessionLocal

CANDIDATES = {
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
    "stepfunctions": "AWS Step Functions",
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


async def main() -> None:
    async with AsyncSessionLocal() as session:
        valid_mappings = {}
        missing_src = []
        missing_tgt = []

        for src, tgt in CANDIDATES.items():
            s_res = await session.execute(
                text("SELECT skill_id, name FROM skills WHERE name = :n"), {"n": src}
            )
            s_row = s_res.fetchone()
            t_res = await session.execute(
                text("SELECT skill_id, name FROM skills WHERE name = :n"), {"n": tgt}
            )
            t_row = t_res.fetchone()

            if not s_row:
                # Check case-insensitive
                s_res = await session.execute(
                    text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:n)"),
                    {"n": src},
                )
                s_row = s_res.fetchone()

            if not t_row:
                # Check case-insensitive
                t_res = await session.execute(
                    text("SELECT skill_id, name FROM skills WHERE LOWER(name) = LOWER(:n)"),
                    {"n": tgt},
                )
                t_row = t_res.fetchone()

            if not s_row:
                missing_src.append(src)
            elif not t_row:
                missing_tgt.append((src, tgt))
            else:
                valid_mappings[s_row[1]] = (s_row[0], t_row[1], t_row[0])

        print(f"VALID MAPPINGS: {len(valid_mappings)}")
        print(f"MISSING SOURCES ({len(missing_src)}): {missing_src}")
        print(f"MISSING TARGETS ({len(missing_tgt)}):")
        for s, t in missing_tgt:
            # Let's search if a similar target exists
            search_pattern = f"%{t.split()[0]}%"
            alt_res = await session.execute(
                text(
                    "SELECT name FROM skills WHERE LOWER(name) LIKE LOWER(:p) AND skill_id IN (SELECT skill_id FROM skill_standards WHERE standard_name = 'Lightcast') LIMIT 5"
                ),
                {"p": search_pattern},
            )
            alts = [r[0] for r in alt_res.fetchall()]
            print(f"   '{s}' -> Target '{t}' NOT FOUND. Lightcast alternatives: {alts}")


if __name__ == "__main__":
    asyncio.run(main())
