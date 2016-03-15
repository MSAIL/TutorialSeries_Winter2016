with open('mcdonalds-normalized-data.tsv') as f:
    text = f.read()
names = '\n'.join(line.split('\t')[-1] for line in text.split('\n'))
minus_last = '\n'.join('\t'.join(line.split('\t')[:-1]) for line in text.split('\n'))
with open('mcdonalds-normalized-data-names.tsv','w') as f:
    f.write(names)
with open('mcdonalds-normalized-data-clean.tsv','w') as f:
    f.write(minus_last)
