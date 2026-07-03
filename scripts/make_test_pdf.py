"""Generate a minimal PDF file with CV text content, using raw PDF syntax."""
import struct

def make_pdf(text):
    # Escape special characters in PDF string
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    
    # Object 1: Catalog
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    # Object 2: Pages
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    # Object 3: Page
    obj3 = (
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    # Object 4: Content stream
    content = (
        b"BT\n"
        b"/F1 11 Tf\n"
        b"50 750 Td\n"
        b"(" + text.encode("latin-1", errors="replace") + b") Tj\n"
        b"ET\n"
    )
    obj4 = b"4 0 obj\n<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream\nendobj\n"
    # Object 5: Font
    obj5 = (
        b"5 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        b"endobj\n"
    )
    
    body = obj1 + obj2 + obj3 + obj4 + obj5
    
    # Cross-reference table
    xref_offset = len(b"%PDF-1.4\n") + 1  # +1 for newline after header
    offsets = []
    offset = 0
    # Calculate offsets for each object
    state = 0
    current_offset = 0
    # Actually, let's build the body first and compute offsets
    # Rebuild with computed offsets
    objects = [obj1, obj2, obj3, obj4, obj5]
    body_parts = []
    xref_offsets = []
    file_len = len(b"%PDF-1.4\n")
    for obj in objects:
        xref_offsets.append(file_len)
        body_parts.append(obj)
        file_len += len(obj)
    
    body = b"".join(body_parts)
    
    xref = b"xref\n"
    xref += b"0 6\n"
    xref += b"0000000000 65535 f \n"
    for off in xref_offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    
    trailer = b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    trailer += b"startxref\n" + str(file_len).encode() + b"\n%%EOF\n"
    
    return b"%PDF-1.4\n" + body + xref + trailer

text = """Senior Software Engineer

SUMMARY
Experienced software engineer with 8+ years building web applications using Python, FastAPI, and cloud technologies.

EXPERIENCE
Senior Engineer, TechCorp (2020-Present)
  - Led development of microservices architecture with Python, FastAPI, and PostgreSQL
  - Designed and implemented RESTful APIs handling 10K+ requests/sec
  - Mentored team of 4 junior engineers

Software Engineer, StartUpX (2017-2020)
  - Built full-stack web applications using React and Django
  - Implemented CI/CD pipelines with Docker and Kubernetes

EDUCATION
B.S. Computer Science, University of Technology (2013-2017)

SKILLS
Python, FastAPI, Django, React, PostgreSQL, Docker, Kubernetes, AWS, Microservices, CI/CD"""

with open("test_cv.pdf", "wb") as f:
    f.write(make_pdf(text))
print("Created test_cv.pdf")
