flowchart LR
    subgraph inputs [Inputs]
        CIF[Crystal structure CIF]
        TAB[Tabular descriptors]
        TXT[Optional text metadata]
    end

    subgraph encoders [Encoders]
        VE[Vision encoder ResNet18]
        TE[Tabular MLP]
        XE[Text encoder optional]
    end

    subgraph fuse [Fusion]
        F[Concat / Gated fusion]
        H[Regression head]
    end

    CIF --> IMG[2D structure image]
    IMG --> VE
    TAB --> TE
    TXT --> XE
    VE --> F
    TE --> F
    XE -.-> F
    F --> H
    H --> Y[Predicted band gap eV]
