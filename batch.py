import os
import json
import pandas as pd
import dsp
import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

tqdm.pandas()
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(".", "local_cache")

from dspy import Models, Example
from src.data_loaders import load_data
from src.programs import InferRetrieveRank

import argparse
import sqlalchemy
from sqlalchemy import text
import pandas as pd
import urllib.parse
from datetime import datetime

num_threads = os.environ.get('DSP_NUM_THREADS', 8)
global engine, schema

def get_db_engine(language: str):
    global engine, schema
    if language == "ko":
        schema = 'query_ko'
        db = {
            "user": "hyjin",
            "password": "Ascent123!@#",
            "host": "analysis002.dev.ascentlab.io",
            "port": 10086,
            "connect_timeout": 180,
        "database": schema,
        }
    elif language == "ja":
        schema = 'query_jp_ja'
        db = {
            "user": "db",
            "password": "Ascent123!@#",
            "host": "analysis004.dev.ascentlab.io",
            "port": 10086,
            "connect_timeout": 180,
        "database": schema,
        }
    
    DB_URL = f"mysql+mysqlconnector://{urllib.parse.quote_plus(db['user'])}:{urllib.parse.quote_plus(db['password'])}@{db['host']}:{db['port']}"
    engine = sqlalchemy.create_engine(DB_URL, pool_size=20,pool_recycle=3600)

def load_depth1_set():
    if language == 'ko':
        with open("/data2/hy.jin/git/xmc.dspy/src/programs/depth_dict.json", "r") as f:
            depth_dict = json.load(f)
    elif language == 'ja':
        with open("/data2/hy.jin/git/xmc.dspy/src/programs/depth_dict_ja.json", "r") as f:
            depth_dict = json.load(f)
        
    depth1_set = set(depth_dict.keys())
    return depth1_set


def load_data():
    get_db_engine(language)
    with engine.connect() as connection:
        # Count the rows before deletion for reporting
        query = text(f"""
            SELECT 
                keyword, predict, volume
            FROM 
                {schema}.llm_entity_topic
                where topic is null
        """)
        result_proxy = connection.execute(query)
        res = result_proxy.fetchall()

    df = pd.DataFrame(res)
    if len(df) == 0:
        print("No data to process")
        exit(0)
        
    df['text'] = df.apply(lambda x: x['keyword'] + " " + (x['predict']), axis=1)

    print(f"Target data size: {len(df)}")
            
    return df
            
def update_progress(pbar, ntotal):
    pbar.set_description(f"Data processed: {ntotal}")
    pbar.update()
    
def insert_dataframe_to_table(df: pd.DataFrame, table_name: str, batch_size: int=100000):
    with engine.connect() as connection:
        for i in range(0, len(df), batch_size):
            if i + batch_size > len(df):
                df.iloc[i:len(df)].to_sql(name=table_name, con=connection, if_exists='append',index=False, schema=schema)
                print(f"Row {i} - {len(df)-1} insert 완료")
            else:
                df.iloc[i:i + batch_size].to_sql(name=table_name, con=connection, if_exists='append',index=False, schema=schema)
                print(f"Row {i} - {i + batch_size} insert 완료")
                
def update_dataframe_to_table(df: pd.DataFrame, table_name: str):
    with engine.connect() as connection:
    
        for _, row in tqdm(df.iterrows()):
            keyword = row['keyword']
            topic = row['topic']
            infer = row['infer']
            retrieved = row['retrieved']
            rerank  = row['rerank']
            
            # topic과 topk를 함께 업데이트
            query = text(f"""
                UPDATE {schema}.{table_name} 
                SET topic = :topic, infer = :infer, retrieved = :retrieved, rerank = :rerank
                WHERE keyword = :keyword
            """)
            
            # query 실행
            connection.execute(query, {"topic": topic, "infer": infer, "retrieved": retrieved, "rerank": rerank, "keyword": keyword})
            
        connection.commit()
        
    print(f"{len(df)} rows updated")

def run_irera(state_path, save_path):
    
    df = load_data()
    texts = df['text'].tolist()
    
    depth1_set = load_depth1_set()
    print(f"Depth1 set: {depth1_set}")
    
    # make dataset
    examples = []
    for idx, text in enumerate(texts):
        example = Example(text=text).with_inputs('text')
        examples.append((idx, example))
    
    # load program
    program = InferRetrieveRank.load(state_path)
    
    def wrapped_program(example_idx, example: Example):
        creating_new_thread = threading.get_ident() not in dsp.settings.stack_by_thread
        if creating_new_thread:
            dsp.settings.stack_by_thread[threading.get_ident()] = list(dsp.settings.main_stack)
        try:
            prediction = program(**example.inputs())
            return example_idx, example, prediction
        except Exception as e:
            print(f"Error for example in dev set: \t\t {example.inputs()}")
            return example_idx, example, dict()
        finally:
            if creating_new_thread:
                del dsp.settings.stack_by_thread[threading.get_ident()]
            
    def _execute_multi_thread(wrapped_program, dataset, num_threads, display_progress=True):
        ntotal = 0
        reordered_dataset = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(wrapped_program, idx, arg) for idx, arg in dataset}
            pbar = tqdm(total=len(dataset), dynamic_ncols=True, disable=not display_progress)

            for future in as_completed(futures):
                example_idx, example, prediction = future.result()
                reordered_dataset.append((example_idx, example, prediction))
                ntotal += 1
                update_progress(pbar, ntotal)
            pbar.close()

        return reordered_dataset, ntotal
    
    # success/failed 폴더 생성
    os.makedirs(f"{save_path}/success", exist_ok=True)
    os.makedirs(f"{save_path}/failed", exist_ok=True)

    # examples를 1000개씩 나눠서 예측하고 결과를 저장
    batch_size = 1000
    for i in range(0, len(examples), batch_size):
        reordered_dataset, _ = _execute_multi_thread(wrapped_program, examples[i:i+batch_size], num_threads)
        results = []
        failed = []
        # 튜플 리스트를 dataframe으로 변환
        for example_idx, example, prediction in reordered_dataset:
            try:
                infer_depth1 = prediction.infer.split(" > ")[0]
                assert infer_depth1 in depth1_set
            except:
                if not prediction:
                    failed.append({
                        "idx": example_idx,
                        "text": example.text,
                        "infer": None,
                        "retrieved": None,
                        "rerank": None,
                        "topic": None,
                    })
                else:
                    failed.append({
                        "idx": example_idx,
                        "text": example.text,
                        "infer": prediction.infer,
                        "retrieved": prediction.retrieved,
                        "rerank": prediction.predictions,
                        "topic": prediction.predictions[0],
                    })
                continue
            results.append({
                "idx": example_idx,
                "text": example.text,
                "infer": prediction.infer,
                "retrieved": prediction.retrieved,
                "rerank": prediction.predictions,
                "topic": prediction.predictions[0],
            })
            
        # dataframe을 csv로 저장
        result_df = pd.DataFrame(results)
        result_df.to_csv(f"{save_path}/success/result_{i}.csv", index=False)
        print(f"Saved result_{i}.csv")
        
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(f"{save_path}/failed/failed_{i}.csv", index=False)
        print(f"Saved failed_{i}.csv")
        
        # dataframe을 df와 join 후 db에 저장
        try:
            result_df = pd.read_csv(f"{save_path}/success/result_{i}.csv")
            df_to_db = pd.merge(result_df, df, on='text', how='left')
            df_to_db.drop(columns=['idx', 'text'], inplace=True)
            df_to_db['collected_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # keyword가 nan인 경우 제외
            df_to_db = df_to_db[~df_to_db['keyword'].isna()]
            update_dataframe_to_table(df_to_db, "llm_entity_topic")
        except:
            print(f"Failed to save to db: {i}")
            continue
    
    return program


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Infer-Retrieve-Rank on an extreme multi-label classification (XMC) dataset."
    )

    # Add arguments
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--state_path", type=str, default="./results/entity_category_infer-retrieve-rank_02/program_state.json")
    parser.add_argument("--save_path", type=str, default="./results_batch/entity_category_infer-retrieve-rank_02")
    parser.add_argument("--lm_config_path", type=str, default="./lm_config.json")

    # Parse the command-line arguments
    args = parser.parse_args()

    language = args.language
    state_path = args.state_path
    lm_config_path = args.lm_config_path
    save_path = args.save_path

    print("state_path: ", state_path)
    print("lm_config_path: ", lm_config_path)
    print("save_path: ", save_path)

    Models(config_path=lm_config_path)

    program = run_irera(state_path, save_path)
    
    # df = load_data(dataset_path)
    
    # # CSV 파일들이 저장된 디렉토리 경로
    # result_path = "./results_batch/entity_category_infer-retrieve-rank_02/success"

    # # 해당 디렉토리에서 모든 CSV 파일 경로를 리스트로 저장
    # csv_files = [os.path.join(result_path, file) for file in os.listdir(result_path) if file.endswith('.csv')]

    # # 각 CSV 파일을 읽어서 데이터프레임으로 변환한 후 리스트에 저장
    # dfs = [pd.read_csv(file) for file in csv_files]

    # # 모든 데이터프레임을 하나로 concat
    # result_df = pd.concat(dfs, ignore_index=True)
    # print(len(result_df))

    # df_to_db = pd.merge(result_df, df, on='text', how='left')
    # df_to_db.drop(columns=['idx', 'text'], inplace=True)
    # df_to_db['collected_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # # keyword가 nan인 경우 제외
    # df_to_db = df_to_db[~df_to_db['keyword'].isna()]
    # df_to_db['keyword'] = df_to_db['keyword'].fillna('NaN')
    # print(len(df_to_db))
    
    # update_dataframe_to_table(df_to_db, "llm_entity_topic")