import PyPDF2
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import fitz

def concat_pdf_text_miner(pdf_path, remove_first=False):

    all_pages = []
    
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        if not remove_first or i > 0:
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        all_pages.append(text_line)
    return all_pages

def is_italic_miner(text):
    length = len(text)
    italic = 0
    for character in text:
        if isinstance(character, LTChar):
            if character.fontname.endswith('ReguItal') :
                italic += 1
    if italic > length/2:
        return True
    else:
        return False
    
def get_color_miner(text) -> str:
    """Font info of LTChar if available, otherwise empty string"""
    for o in text:
        if isinstance(o, LTChar):
            if hasattr(o, 'graphicstate'):
                return f'{o.graphicstate.scolor}'
    return ''

def is_title_miner(text):
    length = len(text)
    big = 0
    for character in text:
        if isinstance(character, LTChar):
            # print(type(character.size))
            if character.size > 11 :
                big += 1
    if big > length/2:
        return True
    else:
        return False   
    

def concat_pdf_text_fitz(pdf_path, remove_first=False):
    
        all_pages = []
        pdf = fitz.open(pdf_path) # filePath is a string that contains the path to the pdf
        for i, page in enumerate(pdf):
            if not remove_first or i > 0:
                dict = page.get_text("dict")
                blocks = dict["blocks"]
                for block in blocks:
                    if "lines" in block.keys():
                        spans = block['lines']
                        for span in spans:
                            all_pages.append(span['spans'])
        # print(all_pages)
        return all_pages


    
    
def extract_many_exams(path_list, funct):
    exams = []
    for path in path_list:
        exams += funct(path)
    return exams

def extract_many_exams_double(path_list_q, path_list_a, funct):
    exams = []
    for path1, path2 in zip(path_list_q, path_list_a):
        exams += funct(path1,path2)
    return exams

def extract_data_crypto_secu (pdf_path) :
    all_pages = concat_pdf_text_miner(pdf_path)
    data = []
    response = ''
    found_quest = False
    found_resp = False
    for text_line in all_pages:
        text = text_line.get_text()
        if not found_quest and not found_resp:
            if text.startswith('Q.'):
                question = text[4:] #remove Q.
                found_quest = True
                continue
                
        if found_quest and not found_resp:
            if is_italic_miner(text_line):
                response = text
                found_resp = True
            else:
                question += " " + text
            continue
        
        if found_quest and found_resp:
            if text.startswith('Q.'):
                found_resp = False
                found_quest = False
                data.append({'question': question, 'answer': response, "course" : "Cryptography and Security" })
                question = text[4:]
                found_quest = True
            else :
                response += " " + text  
            continue
    
    data.append({'question': question, 'answer': response, "course" : "ML" })
    return data

def extract_data_info_secu_priva (pdf_path) :
    all_pages_q = concat_pdf_text_miner(pdf_path)
    data_q = {}
    data_r = {}
    solution = False
    for text_line in all_pages_q:
        text = text_line.get_text()
        
        if text.startswith('Solutions to the Exercises'):
            solution = True
            continue
        
        if text.startswith('Exercise '):
            exo_nb = text.split(' ')[1]
            data_q[exo_nb] = ''
            continue
            
        
        elif text.startswith('Solution '):
            exo_nb = text.split(' ')[1]
            data_r[exo_nb] = ''
            continue
        
        if data_q == {}:
            continue
            
        if solution :
            data_r[exo_nb] += text + ' '
        else : 
            data_q[exo_nb] += text+ ' '
    

    data = [{"question" : data_q[exo], "answer" : data_r[exo], "course" : "Information security and privacy"} for exo in data_q.keys()]

    return data 

def extract_data_ope_sys (pdf_path) :
    all_pages = concat_pdf_text_miner(pdf_path)
    data = []
    response = ''
    found_quest = False
    found_resp = False
    for text_line in all_pages:
        text = text_line.get_text()
        
        if not found_quest and not found_resp:
            if text.startswith('(') and text[2] == ')':
                question = text[4:]
                found_quest = True
            continue
        
        if found_quest and not found_resp:

            if get_color_miner(text_line) == '(1, 0, 0)':
                found_resp = True
                response = text
            else :
                question += " " + text
            continue
            
        if found_quest and found_resp:
            if text.startswith('('):
                data.append({'question': question, 'answer': response, "course" : "Introduction to operating systems" })
                question = text[4:]
                found_resp = False
                found_quest = True
                
            elif get_color_miner(text_line) == '(1, 0, 0)':
                response += " " + text
            continue
    data.append({'question': question, 'answer': response, "course" : "ML" })
    return data

import fitz
import os

def extract_data_ml (pdf_path) :
    all_pages = concat_pdf_text_fitz(pdf_path, remove_first=True)
    # print(all_pages)
    data = []
    response = ''
    search_consign1 = False
    search_consign2 = False
    found_quest = False
    found_resp = False
    prop = False
    i = 0
    
    
    for block in all_pages:
        i +=1
        j = i-1
        for text_line in block:
            j += +1
            text = text_line["text"]
            # print(text_line)
            
            if text_line["size"] > 11.9 and text.startswith('Third part'):
                if response.endswith("y y") :
                        response = response[:-7]
                data.append({'question': question, 'answer': response, "course" : "ML" })
                return data
            if text.startswith('For your examination') or text.startswith('multiple-choice') or text.startswith('+1/') or text == "DRAFT":
                continue
            
            if 12 > text_line["size"] > 11.9 and not found_quest and not found_resp:
                if text.startswith("Second part:") or text.startswith("First part:") :
                    consign1 = ""
                    # print("consign1", text)
                    search_consign1 = True
                    continue
                else : 
                    question = text
                    # print("title", question)
                    search_consign1 = False
                    consign2 = ""
                    search_consign2 = True
                    continue
                    
            if search_consign1 :
                if text.startswith('Question '):
                    # print("question", text)
                    question = text[11:] #remove Q.
                    search_consign2 = False
                    search_consign1 = False
                    found_quest = True
                    continue   
                else : 
                    text = text.replace("each", "the")
                    text = text.replace("Each", "the")
                    consign1 += " " + text
                    # print("consign1", consign1)
                    continue   
                
            if search_consign2 :
                if text.startswith('Question '):
                    # print("question", text)
                    question = text[11:] #remove Q.
                    search_consign2 = False
                    search_consign1 = False
                    found_quest = True
                    continue                  
                else :
                    consign2 += " " + text
                    # print("consign2", consign2)
                    continue  
                         
                    
            if found_quest and not found_resp:
                if text.startswith('Solution:'):
                    # print("solution", text)
                    response = text[11:]
                    found_resp = True    
                    prop = False 
                    continue         
                else :
                    if question != "" :
                        if question[-1] == "?" and len(text) <10 :
                            prop = True
                        if prop and i ==j and not question.endswith("√"): 
                            text = "\n[] " + text
                    # print("suite quest",text)
                    question += " " + text
                    continue
            
            if found_quest and found_resp:
                if text.startswith('Question '):
                    data.append({'question': consign1 +consign2 +question, 'answer': response, "course" : "ML" })
                    question = text[11:]
                    # print("question", text)
                    found_quest = True
                    found_resp = False
                    continue
                if 12 > text_line["size"] > 11.9 :
                    if response.endswith("y y") :
                        response = response[:-7]
                    data.append({'question': consign1 +consign2 +question, 'answer': response, "course" : "ML" })
                    if text.startswith("Second part:") or text.startswith("First part:") :
                        consign1 = ""
                        consign2 = ""
                        # print("consign1", text)
                        search_consign1 = True
                        found_quest = False
                        found_resp = False
                        continue
                    else : 
                        question = text
                        # print("title", question)
                        found_quest = False
                        found_resp = False
                        search_consign1 = False
                        consign2 = ""
                        search_consign2 = True
                        continue
                else :
                    # print("solution2", text)
                    response += " " + text  
                    continue
    data.append({'question': question, 'answer': response, "course" : "ML" })
    return data



def extract_data_ml2 (pdf_path) :
    all_pages = concat_pdf_text_fitz(pdf_path, remove_first=True)
    data = []
    response = ''
    found_quest = False
    found_resp = False
    prop = False
    i = 0
    
    for block in all_pages:
        i +=1
        j = i-1
        for text_line in block:
            j += +1
            text = text_line["text"]
            # print(text)
            
            if text.startswith('+1/') or text.endswith(" point]") or text.endswith(" points]"):
                continue
            
            if not found_quest and text.startswith('Problem '):
                # print("question", text)
                question = ""
                found_quest = True
                continue   
                         
            if found_quest and not found_resp:
                if text.startswith('Solution'):
                    # print("solution", text)
                    response = text[11:]
                    found_resp = True    
                    prop = False 
                    continue         
                else :
                    if question != "" :
                        if question[-1] == "?" and len(text) <10 or len(text) <5:
                            prop = True
                        if prop and i ==j and not question.endswith("√"): 
                            text = "\n[] " + text
                    # print("suite quest",text)
                    question += " " + text
                    continue
            
            if found_quest and found_resp:
                if text.startswith('Problem '):
                    data.append({'question': question, 'answer': response, "course" : "ML" })
                    question = text[20:]
                    # print("questionn", text)
                    found_quest = True
                    found_resp = False
                    continue
                else :
                    # print("solution2", text)
                    response += " " + text  
                    continue
    data.append({'question': question, 'answer': response, "course" : "ML" })
    return data



def extract_data_ph (pdf_path_q, pdf_path_a) :
    if pdf_path_q.startswith("z") :
        all_pages_q = concat_pdf_text_fitz(pdf_path_q, remove_first=True)
    else :
        all_pages_q = concat_pdf_text_fitz(pdf_path_q)
    all_pages_a = concat_pdf_text_fitz(pdf_path_a)
    # print(pdf_path_a, pdf_path_q)
    
    data_q = {}
    data_a = {}
    conceptu = False
    start = False
    found_quest = False
    found_resp = False
    for block in all_pages_q:
        for text_line in block:
            text = text_line["text"]
            
            if text.startswith('PHYS-101'):
                start = True
                continue
        
            if start : 
                
                if text.endswith("points)") or text.startswith("20 January") or text.startswith("31 Oct") or text.startswith("5 Dec") or text.startswith("19 Jan") :
                    continue
                
                if (text_line["font"]== "SFBX1000" or text_line["font"]=="CMBX12"):
                    if text.startswith("1") or text.startswith("2") or text.startswith("3") or text.startswith("4") or text.startswith("5") or text.startswith("6") or text.startswith("7") or text.startswith("8") or text.startswith("9") or text.startswith("0") :
                        # print("title", text)
                        found_quest = True
                        exo_title = text[:6]
                        if "Review:" in text :
                            # print("-conceptu")
                            conceptu = True  
                            data_q[exo_title.lower()] = "" 
                        else :
                            data_q[exo_title.lower()] = "" 
                            conceptu = False       
                        continue
                
                
                if found_quest :
                    # print("question", text)
                    if conceptu and (text.startswith('a.') or text.startswith('b.') or text.startswith("c.")) :
                        # print("-conceptu")
                        exo_title = exo_title + " " + text[0]
                        data_q[exo_title.lower()] = ""
                    elif text.startswith("b."):
                        # print("-b")
                        found_quest = False
                        continue
                    else :
                        # print("-else")
                        data_q[exo_title.lower()] += text +" "
                        continue
    start = False
    for block in all_pages_a:
        for text_line in block:
            text = text_line["text"]
            
            if text.startswith('PHYS-101'):
                start = True
                continue
        
            if start : 
                # bold ?
                if (text_line["font"]== "SFBX1000" or text_line["font"]=="CMBX12") and not text.endswith("points"):
                    if text.startswith("1") or text.startswith("2") or text.startswith("3") or text.startswith("4") or text.startswith("5") or text.startswith("6") or text.startswith("7") or text.startswith("8") or text.startswith("9") or text.startswith("0") :
                        # print("title", text)
                        exo_title = text[:6]
                        data_a[exo_title.lower()] = "" 
                        found_resp = True
                        if "Review:" in text :
                            # print("-conceptu")
                            conceptu = True  
                        else :
                            conceptu = False       
                        continue
                
                
                if found_resp :
                    # print("question", text)
                    # print("title", exo_title)
                    if conceptu and (text.startswith('a.') or text.startswith('b.') or text.startswith("c.")) :
                        # print("-conceptu2")
                        exo_title = exo_title + " " + text[0]
                        data_a[exo_title.lower()] = ""
                        continue
                    elif text.startswith("b."):
                        # print("-b")
                        found_resp = False
                        continue
                    else :
                        # print("-else")
                        data_a[exo_title.lower()] +=  text +" "
                        continue              
    # print(list(data_q.keys()))
    # print(list(data_a.keys())) 
    data = [{"question" : data_q[exo], "answer": data_a[exo], "course" : "Physique meca"} for exo in data_q.keys()]
    return data 