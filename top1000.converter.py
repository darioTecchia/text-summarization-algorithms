'''
This script convert the ScisummNet top1000 dataset in a format compatible with
the datasets used in the mine experiments.
The dataset can be downloaded from here
https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip
'''
import xml.etree.ElementTree as ET
import os
import pandas
import html

result = dict()
result['original_texts'] = []
result['human_summaries'] = []

dirs = os.listdir('top1000_complete')

for index, dir in enumerate(dirs):
    print(dir, str(index) + '/' + str(len(dirs)))
    root = ET.parse('top1000_complete/' + dir + '/Documents_xml/' + dir + '.xml').getroot()
    text = ' '.join([x.text for x in root.findall("SECTION/S")])
    with open('top1000_complete/' + dir + '/summary/' + dir + '.gold.txt') as f:
        summary = f.read().splitlines()
    summary = ' '.join(summary[1:])
    text = text.replace(',', '')
    summary = summary.replace(',', '')
    if(text == '' or summary == ''): continue
    result['original_texts'].append(html.unescape(text))
    result['human_summaries'].append(html.unescape(summary))

pandas.DataFrame(result).to_csv('datasets/news.csv', index=False, sep=',')