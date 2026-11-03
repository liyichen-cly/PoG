import ast
from prompt_list import *
import json
import time
import openai
import re
import requests
import random
from prompt_list import *
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import os

color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo", print_in=True, print_out=True):
    if print_in:
        print(color_green+prompt+color_end)

    if 'gpt' in engine:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
        message_prompt = {"role":"user","content":prompt}
        messages.append(message_prompt)
        client = openai.OpenAI(api_key=opeani_api_keys)
        completion = client.chat.completions.create(
                model=engine,
                messages = messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0)

        result = completion.choices[0].message.content

        token_num = {"total": completion.usage.total_tokens, "input": completion.usage.prompt_tokens, "output": completion.usage.completion_tokens}

        if print_out:
            print(color_yellow + result + color_end)
        return result, token_num


def convert_dict_name(ent_rel_ent_dict, entid_name):
    name_dict = {}
    for topic_e, h_t_dict in ent_rel_ent_dict.items():
        if entid_name[topic_e] not in name_dict.keys():
            name_dict[entid_name[topic_e]] = {}

        for h_t, r_e_dict in h_t_dict.items():
            if h_t not in name_dict[entid_name[topic_e]].keys():
                name_dict[entid_name[topic_e]][h_t] = {}
            
            for rela, e_list in r_e_dict.items():
                if rela not in name_dict[entid_name[topic_e]][h_t].keys():
                    name_dict[entid_name[topic_e]][h_t][rela] = []
                for ent in e_list:
                    if entid_name[ent] not in name_dict[entid_name[topic_e]][h_t][rela]:
                        name_dict[entid_name[topic_e]][h_t][rela].append(entid_name[ent])
    return name_dict

    

def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name):
    tt = time.time()-start_time
    dict = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "call_num": call_num, "total_token": all_t['total'], "input_token": all_t['input'], "output_token": all_t['output'], "time": tt}
    with open("PoG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

def extract_add_ent(string):
    first_brace_p = string.find('[')
    last_brace_p = string.rfind(']')
    string = string[first_brace_p:last_brace_p+1]
    try:
        new_string = ast.literal_eval(string)
    except:
        s_list = string.split('\', \'')
        if len(s_list) == 1:
            new_string = [s_list[0].strip('[\'').strip('\']')]
        else:
            new_string = [s.strip('[\'').strip('\']') for s in s_list]
    return new_string

def extract_memory(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    return string

def extract_reason_and_anwer(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    answer = re.search(r'"Answer":\s*"(.*?)"', string)
    if answer:
        answer = answer.group(1)
    else:
        answer = re.search(r'"Answer":\s*(\[[^\]]+\])', string).group(1)

    reason = re.search(r'"R":\s*"(.*?)"', string).group(1)
    sufficient = re.search(r'"Sufficient":\s*"(.*?)"', string).group(1)
    print("Answer:", answer)
    print("Reason:", reason)
    print("Sufficient:", sufficient)
    return answer, reason, sufficient

def extract_add_and_reason(string):
    first_brace_p = string.find('{')
    last_brace_p = string.rfind('}')
    string = string[first_brace_p:last_brace_p+1]
    flag = re.search(r'"Add":\s*"(.*?)"', string).group(1)
    reason = re.search(r'"Reason":\s*"(.*?)"', string).group(1)

    print("Add:", flag)
    print("Reason:", reason)
    if 'yes' in flag.lower():
        return True, reason
    else:
        return False, reason
    


def generate_without_explored_paths(question, subquestions, args):
    prompt = cot_prompt + question 

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    return response, token_num

def break_question(question, args): 
    prompt = subobjective_prompt + question
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    first_brace_p = response.find('[')
    last_brace_p = response.rfind(']')
    response = response[first_brace_p:last_brace_p+1]
    return response, token_num

def get_subquestions(q_mem_f_path, question, args):
    sub_questions, token_num = break_question(question, args)
    with open(q_mem_f_path+'/'+'subq', 'w', encoding='utf-8') as f:
        f.write(str(sub_questions))

    return sub_questions, token_num

def if_finish_list(question, lst, depth_ent_rel_ent_dict, entid_name, name_entid, q_mem_f_path, results, cluster_chain_of_entities, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    if all(elem == "[FINISH_ID]" for elem in lst):
        new_lst = []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
    
    all_ent_set = set()
    for dep, ent_rel_ent_dict in depth_ent_rel_ent_dict.items():
        for topic_e, h_t_dict in ent_rel_ent_dict.items():
            all_ent_set.add(topic_e)
            for h_t, r_e_dict in h_t_dict.items():
                for rela, e_list in r_e_dict.items():
                    if all(entid_name[item].startswith('m.') for item in e_list) and len(e_list)>10:
                        e_list = random.sample(e_list, 10)
                        
                    if len(e_list) > 70:
                        print('··········exceed 70 entities··········')
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        print('sentence:', topn_entities)
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        all_ent_set |= (set(e_list))

    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    
    prompt = judge_reverse+question+'\nEntities set to be retrieved: ' + str(list(set(sorted([entid_name[ent_i] for ent_i in new_lst])))) +'\nMemory: '+his_mem+'\nKnowledge Triplets:'+chain_prompt

    cur_call_time += 1
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    for kk in token_num.keys():
        cur_token[kk] += token_num[kk]

    flag, reason = extract_add_and_reason(response)

    if flag:
        other_entities = sorted(list(all_ent_set - set(new_lst)))
        other_entities_name = [entid_name[ent_i] for ent_i in other_entities]
        
        print('filter already', [entid_name[ent_i] for ent_i in new_lst], [entid_name[ent_i] for ent_i in all_ent_set], other_entities_name)

        prompt = add_ent_prompt+question+'\nReason: '+reason+'\nCandidate Entities: ' + str(sorted(other_entities_name))+'\nMemory: '+his_mem

        cur_call_time += 1
        response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

        for kk in token_num.keys():
            cur_token[kk] += token_num[kk]

        add_ent_list = extract_add_ent(response)
        add_ent_list = [name_entid[ent_i] for ent_i in add_ent_list if ent_i in other_entities_name]
        add_ent_list = sorted(add_ent_list)
        if add_ent_list:
            print('add reverse ent:', len(add_ent_list), [entid_name[ent_i] for ent_i in add_ent_list])
            return new_lst, add_ent_list, cur_call_time, cur_token
    return new_lst, [], cur_call_time, cur_token

    
def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa}.")
        exit(-1)
    return datas, question_string

