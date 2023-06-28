import openai
import re

# openai.organization = 'Personal'
openai.api_key = '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'

def Ask_ChatGPT(message):

    # 応答設定
    completion = openai.ChatCompletion.create(
                 model    = "gpt-3.5-turbo",     # モデルを選択
                 messages = [{
                            "role":"user",       # 役割
                            "content":message,   # メッセージ 
                            }],

                 max_tokens  = 1024,             # 生成する文章の最大単語数
                 n           = 1,                # いくつの返答を生成するか
                 stop        = None,             # 指定した単語が出現した場合、文章生成を打ち切る
                 temperature = 0.5,              # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
    )

    # 応答
    response = completion.choices[0].message.content

    # 応答内容出力
    return response


def is_more_vulnerable(item1, item2):
    prompt1_en = 'When filling a basket, contact between items may cause damage to the items and damage the value of the product. Now place the items on either of the following. I want to place it so that the value of the product is not damaged as much as possible. Which of the following should I put it on?\n'\
                 'Example"""\n'\
                 'input: 1. strawberry, 2. apple\n'\
                 'output: 1\n'\
                 '"""\n'\
                 '\n'\
                 f'input: 1. {item1}, 2. {item2}\n'\
                 'output:\n'
    #return prompt1_en
    return Ask_ChatGPT(prompt1_en)



prompt1_jp = 'カゴにものを詰めるとき，物品同士の接触により物品が傷み，商品価値が損なわれることがあります．今から以下のどちらかの上に物を置きます．できるだけ商品価値が損なわれないように置きたいです．どちらの上に置くのが良いですか？'\
             '例"""'\
             'input:  1. いちご, 2. りんご'\
             'output: 1'\
             '"""'\
             ''\
             'input: 1.ルマンド（菓子袋）, 2. 寿司パック'\
             'output:'

def is_easily_crushed(item1, item2):
    prompt = '例"""\n'\
             'input: 1.お菓子袋，2.豆腐パック の2つを比べると，どちらが潰れやすいですか？\n'\
             'output: 1\n'\
             '"""\n'\
             '\n'\
             f'input: 1.{item1}, 2.{item2} の2つを比べると，どちらが潰れやすいですか？\n'\
             'output:\n'
    print(prompt)
    return Ask_ChatGPT(prompt)

def put_on_1_or_2(item0, item1, item2):
    prompt = 'スーパーマーケットで買った物をカゴに詰めています．\n'\
             f'Aさんは{item0}を{item1}の上に置きました．\n'\
             f'Bさんは{item0}を{item2}の上に置きました．\n'\
             'AさんとBさんのどちらの判断が優れていますか？\n'\
             f'{item1}及び{item2}の損傷の少なさという観点で判断してください．\n'
    print(prompt)
    response = Ask_ChatGPT(prompt)
    print(response)
    A_or_B = re.search('[AB].*優れて', response).group(0)[0]
    return (0, item1) if A_or_B == 'A' else (1, item2)

# response = '物品の損傷の少なさという観点から判断すると、Aさんの判断が優れています。ツナ缶は一般的に頑丈で、パッケージがしっかりしているため、寿司パックの上に置いても損傷する可能性は比較的低いです。一方、りんごは比較的脆弱な果物であり、ツナ缶の重みや形状によっては損傷する可能性が高くなります。したがって、ツナ缶を寿司パックの上に置く方が、ツナ缶自体の損傷を最小限に抑えることができるでしょう。ただし、他の商品やパッケージの状態、ツナ缶の大きさや形状によっては、他の配置方法が適している場合もあります。'




