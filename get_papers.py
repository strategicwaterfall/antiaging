
#get papers for all companies using pymed library - and then rank order them to collect top 10 papers fro each company
import pandas as pd
import regex as re
#%pip install pymed
import numpy as np
from pymed import PubMed
pubmed_api_key = 'd3eec1f6d6ae8958046746ab8324ab97c608'


pubmed = PubMed(tool='PubMedSearcher', email='pari2798@gmail.com')
my_api_key = pubmed_api_key
pubmed.parameters.update({'api_key': my_api_key})

def get_papers_of_company(company_name):
    """ function that takes a company name and returns a list of tuples of all the papers for that company"""
    article_list = []
    try:


        company_name = company_name.lower()
        query_new = "{0}[Affiliation]".format(company_name)
        results = pubmed.query(query_new, max_results=500)
        article_list = []
        
        
        for article in results:
            try:
                article_id = article.pubmed_id.partition('\n')[0]
                title = article.title
                doi = article.doi
                authors = article.authors
                abstract = article.abstract
                journal = article.journal
                publication_date = article.publication_date
                try:

                # Extract and format information from the article
    
                    if article.keywords:
                        if None in article.keywords:
                            article.keywords.remove(None)
                        keywords = '", "'.join(article.keywords)
                    tuple_article = (company_name,article_id,title,keywords,publication_date,abstract,journal,doi,publication_date,authors)

                except:
                    keywords = ["none available"]
                    
                    tuple_article = (company_name,article_id,title,keywords,publication_date,abstract,journal,doi,publication_date,authors)
                article_list.append(tuple_article)
            except:
                pass
            
            
    except:
        article_list.append((company_name,'None','None','None','None','None','None','None','None','None'))
    return article_list


def convert_list_of_tuples_to_dataframe(list_of_tuples):
    """ convert a list of tuples into a dataframe"""
    df = pd.DataFrame(list_of_tuples, columns=['company_name','article_id','title','keywords','publication_date','abstract','journal','doi','publication_date','authors'])
    return df


def get_papers_for_all_companies(list_of_companies):
    
    """ function that takes a list of companies and returns a dataframe of all the papers for each company"""           
    list_of_dataframes = []
    for company in list_of_companies:
        list_of_tuples = get_papers_of_company(company)
        df = convert_list_of_tuples_to_dataframe(list_of_tuples)
        list_of_dataframes.append(df)
        
    return (pd.concat(list_of_dataframes)).reset_index(drop=True)
# clean author names from a list of json or dicts to dataframne columns 

""""clean author names from a list of json or dicts to dataframne columns 
"""




if __name__ == "__main__":
    comapny_list = ['Fauna Bio',
    'Fountain Therapeutics',
    'IntraClear Biologics',
    'GenFlow Biosciences',
    'Equator Therapeutics']
    get_papers_for_all_companies(comapny_list).to_csv('company_papers.csv',index=False)