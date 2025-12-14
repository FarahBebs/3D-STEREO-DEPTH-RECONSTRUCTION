#!/usr/bin/env python3
"""
Convert Markdown report to PDF using reportlab
"""

import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os
import re


def markdown_to_pdf(markdown_file, pdf_file):
    """Convert markdown file to PDF"""

    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content, extensions=['tables', 'fenced_code'])

    # Create PDF document
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=15,
        fontName='Helvetica-Bold'
    )

    heading3_style = ParagraphStyle(
        'Heading3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=10,
        fontName='Helvetica'
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=10,
        fontName='Courier',
        backColor='#f0f0f0',
        borderPadding=5,
        spaceAfter=10
    )

    # Split content into sections
    story = []

    # Process HTML content
    sections = re.split(r'<h([1-6])>(.*?)</h\1>', html_content)

    for i in range(1, len(sections), 3):
        level = int(sections[i])
        title = re.sub(r'<[^>]+>', '', sections[i+1])  # Remove HTML tags
        content = sections[i+2] if i+2 < len(sections) else ""

        # Add heading
        if level == 1:
            story.append(Paragraph(title, title_style))
        elif level == 2:
            story.append(Paragraph(title, heading1_style))
        elif level == 3:
            story.append(Paragraph(title, heading2_style))
        elif level >= 4:
            story.append(Paragraph(title, heading3_style))

        story.append(Spacer(1, 12))

        # Process content
        if content:
            # Split by paragraphs and other elements
            paragraphs = re.split(r'</?p>', content)
            for para in paragraphs:
                if para.strip():
                    # Check for code blocks
                    if '<code>' in para or '<pre>' in para:
                        # Extract code content
                        code_match = re.search(
                            r'<pre><code>(.*?)</code></pre>', para, re.DOTALL)
                        if code_match:
                            code_text = code_match.group(1)
                            story.append(Paragraph(code_text, code_style))
                        else:
                            # Inline code
                            para = re.sub(r'<code>(.*?)</code>',
                                          r'<font face="Courier">\1</font>', para)
                            story.append(Paragraph(para, normal_style))
                    elif '<img' in para:
                        # Handle images
                        img_match = re.search(
                            r'<img[^>]+src="([^"]+)"[^>]*>', para)
                        if img_match:
                            img_path = img_match.group(1)
                            if os.path.exists(img_path):
                                try:
                                    img = Image(img_path, width=6 *
                                                inch, height=4*inch)
                                    story.append(img)
                                    story.append(Spacer(1, 12))
                                except:
                                    story.append(
                                        Paragraph(f"[Image: {img_path}]", normal_style))
                    elif '<table>' in para:
                        # Handle tables (simplified)
                        story.append(
                            Paragraph("[Table content - see markdown for details]", normal_style))
                    else:
                        # Clean HTML tags for regular paragraphs
                        clean_para = re.sub(r'<[^>]+>', '', para).strip()
                        if clean_para:
                            # Handle math expressions (simplified)
                            clean_para = clean_para.replace(
                                '\\(', '$').replace('\\)', '$')
                            clean_para = clean_para.replace(
                                '\\[', '$$').replace('\\]', '$$')
                            story.append(Paragraph(clean_para, normal_style))

        story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    print(f"PDF created: {pdf_file}")


if __name__ == "__main__":
    markdown_to_pdf("academic_report.md", "academic_report.pdf")
