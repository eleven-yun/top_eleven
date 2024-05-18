# top_eleven

## Objective
This project aims to learn a model that predicts the outcome of a football game, based on information of the two teams that are going to play the game.

```mermaid
graph LR;
    subgraph Input
        team1[Team1 Info];
        team2[Team2 Info];
        game((Game Info));
    end
    subgraph Model
        model[(TOP ELEVEN)];
    end
    subgraph Output
        pred((Prediction));
        team1_win[A: Team1 Win];
        team2_win[B: Team2 Win];
        draw[C: A Draw];
    end
    team1-->game;
    team2-->game;
    game-->model;
    model-->pred;
    pred--P(A)-->team1_win;
    pred--P(B)-->team2_win;
    pred--P(C)-->draw;
```

## Problem Formulation

This is essentially a multi-class classification problem, where the number of classes $K=3$.

The expected output of the model is the predicted probability for each class:
* $P(A) \in [0, 1]$: The probability for event $A$ that *Team1* wins the game.
* $P(B) \in [0, 1]$: The probability for event $B$ that *Team2* wins the game.
* $P(C) \in [0, 1]$: The probability for event $C$ that the game ends in a draw.

Note that $P(A) + P(B) + P(C) = 1$.

## The Model

The model is a standard Transformer which follows the general encoder-decoder framework.

```mermaid
graph LR
    subgraph Input
        subgraph Game
            history(["History"]);
            subgraph Team1
                history1(["History"]);
                coach1(["Coach"]);
                players1(["Players"]);
                coach1-.->players1;
            end
            subgraph Team2
                history2(["History"]);
                coach2(["Coach"]);
                players2(["Players"]);
                coach2-.->players2;
            end
            ref(["Referee"]);
        end
    end

    subgraph Embedding
        ge("Game-Level\nEmbedding");
        gpe("Positional\nEmbedding");
        ie("Individual-Level\nEmbedding");
        ipe("Positional\nEmbedding");
        gadd(("Add"));
        iadd(("Add"));
    end

    subgraph Model
        subgraph Nx Encoder
            enc_sa("Self\nAttention");
            enc_mlp("MLP");
            enc_sa-->enc_mlp;
        end
        subgraph Nx Decoder
            dec_sa("Self\nAttention");
            dec_xa("Cross\nAttention");
            dec_mlp("MLP");
            dec_sa--"Query"-->dec_xa;
            dec_xa-->dec_mlp;
        end
        enc_mlp--"Key, Value"-->dec_xa;
        softmax("Softmax");
        linear("Linear");
        dec_mlp-->linear;
        linear-->softmax;
    end

    subgraph Output
        pred(["Prediction"]);
    end

    history--"Last M Games:\nTeam1 vs Team2"-->ge;
    history1--"Last M Games:\nTeam1 vs Team?"-->ge;
    history2--"Last M Games:\nTeam2 vs Team?"-->ge;

    gpe--"Time, Place"-->gadd;
    ge-->gadd;

    gadd-->dec_sa;

    coach1-->ie;
    coach2-->ie;
    ref-->ie;
    players1--"Lineup:\n11 Players"-->ie;
    players2--"Lineup:\n11 Players"-->ie;

    ipe--"Formation"-->iadd;
    ie-->iadd;

    iadd-->enc_sa;

    softmax--"Probablity"-->pred;
```