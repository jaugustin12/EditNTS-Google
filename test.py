# import fkgl

# with open('new_data/document-aligned.v2/simple-processed.txt',encoding="utf8") as f:
#     contents = f.readlines()
#     out_string = ""
#     for phrase in contents:
#         out_string += phrase + '\n'
#     print(fkgl.Readability(out_string).FleschKincaidGradeLevel())

with open('data/raw/newsela/newsela.comp.txt') as f:
    contents1 = f.readlines()
    contents1_lower = [sentence.lower() for sentence in contents1]

with open('data/processed/newsela/output/all-newsela209') as g:
    contents2 = g.readlines()
    contents2_lower = [sentence.lower() for sentence in contents2]

diff = 0
for i in range(len(contents1)):
    if contents1_lower[i] != contents2_lower[i]:
        diff +=1

print(diff, len(contents1))
