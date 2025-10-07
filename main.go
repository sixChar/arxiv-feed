package main

import (
    "bytes"
    "context"
    "errors"
    "flag"
    "fmt"
    "html/template"
    "log"
    "os"
    "io"
    "encoding/xml"
    "encoding/json"
    "net/http"
    "net/url"
    "regexp"
    "strconv"
    "strings"
    "time"

    "database/sql"
    sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
    sqlite3 "github.com/mattn/go-sqlite3"

    "golang.org/x/crypto/bcrypt"

    "github.com/golang-jwt/jwt/v5"
    "github.com/joho/godotenv"
)


const ARXIV_RSS_URL = "https://rss.arxiv.org/rss/"

const RSS_TIME_LAYOUT_STR = "Mon, 02 Jan 2006 15:04:05 -0700"
const ISO_86O1_TIME_LAYOUT_STR = "2006-01-02 15:04:05"


const EMBED_MODEL = "nomic-embed-text:v1.5"
const EMBED_URL = "http://127.0.0.1:11434/api/embed"
const EMBED_DIM = 768

// Chunk size to use for embedding large numbers at a time
const CHUNK_SIZE = 32

var JWT_KEY []byte
var TOGETHER_TOKEN string

const USER_ID_CONTEXT_KEY = "userId"
const BCRYPT_STRENGTH = 14


type OAI struct {
    XMLName      xml.Name   `xml:"OAI-PMH"`
    ResponseDate string     `xml:"responseDate"` 
    Request      OAIRequest `xml:"request"`
    ListSets     *ListSets  `xml:"ListSets"`
    Error        *OAIError  `xml:"error"`        // OAI-PMH can return <error> instead
}

type OAIRequest struct {
    Verb string `xml:"verb,attr"`
    URL  string `xml:",chardata"`
}

type ListSets struct {
    Sets            []Set  `xml:"set"`
    ResumptionToken string `xml:"resumptionToken"`
}

type Set struct {
    Spec string `xml:"setSpec"`
    Name string `xml:"setName"`
}

type OAIError struct {
    Code string `xml:"code,attr"`
    Text string `xml:",chardata"`
}


type ArxSubject struct {
    Name string
    DotPath string
    IsTop bool
    Parent string
}


type AtomLink struct {
	Href string `xml:"href,attr"`
	Rel  string `xml:"rel,attr"`
	Type string `xml:"type,attr"`
}

type GUID struct {
	Value        string `xml:",chardata"`
	IsPermaLink  bool   `xml:"isPermaLink,attr"`
}

type RSSItem struct {
	Title        string   `xml:"title"`
	Link         string   `xml:"link"`
	Description  string   `xml:"description"`
	GUID         GUID     `xml:"guid"`
	Categories   []string `xml:"category"`
	PubDate      string   `xml:"pubDate"`
	AnnounceType string   `xml:"http://arxiv.org/schemas/atom announce_type"`
	DCCreator    string   `xml:"http://purl.org/dc/elements/1.1/ creator"`
}

type RSS struct {
	XMLName xml.Name   `xml:"rss"`
	Channel RSSChannel `xml:"channel"`
}

type RSSChannel struct {
	Title         string    `xml:"title"`
	Link          string    `xml:"link"`
	Desc          string    `xml:"description"`
	AtomSelfLink  AtomLink  `xml:"http://www.w3.org/2005/Atom link"` // <atom:link .../>
	Docs          string    `xml:"docs"`
	Lang          string    `xml:"language"`
	LastBuildDate string    `xml:"lastBuildDate"`
	Editor        string    `xml:"managingEditor"`
	PubDate       string    `xml:"pubDate"`
	SkipDays      []string  `xml:"skipDays>day"`
	Items         []RSSItem `xml:"item"`
}


type PaperResult struct {
    ArxivId string
    Title string
    Desc string
    PubDate string
    Score float32
}

type EmbeddingRequest struct {
    Model string `json:"model"`
    Input []string `json:"input"`
}

func ollamaEmbed(toEmbed []string) ([][EMBED_DIM]float32, error) {
    toEmbedNomic := make([]string, len(toEmbed))
    prefix := "clustering: "

    for i,str := range toEmbed {
        toEmbedNomic[i] = prefix + str
    }

    reqBodyStruct := EmbeddingRequest{
        Model: EMBED_MODEL,
        Input: toEmbedNomic,
    }

    reqBody, err := json.Marshal(reqBodyStruct)
    if err != nil {
        return nil, err
    }

    req, err := http.NewRequest("POST", EMBED_URL, bytes.NewReader(reqBody))
    if err != nil {
        return nil, err
    }

    req.Header.Set("Content-Type", "application/json")
    client := http.Client{Timeout: 5 * time.Minute}


    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    respBody, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    if (resp.StatusCode != http.StatusOK) {
        err := fmt.Errorf("Request error in ollamaEmbed. Got status: %s\n%s\n", resp.StatusCode, respBody)
        return nil, err
    }


    res := struct {
        Model           string                  `json:"model"`
        Embeddings      [][EMBED_DIM]float32    `json:"embeddings"`
        TotalDuration   uint64                  `json:"total_duration"`
        LoadDuration    uint64                  `json:"load_duration"`
        PromptEvalCount uint64                  `json:"prompt_eval_count"`
    }{}


    err = json.Unmarshal(respBody, &res)
    if err != nil {
        return nil, err
    }

    
    return res.Embeddings, nil
}


type TooManyInputTokensError struct {
    MaxTokens    int
    InputTokens int
}

func (e *TooManyInputTokensError) Error() string {
    return fmt.Sprintf("Input has too many tokens. Context length is %d but input had %d tokens.", e.MaxTokens, e.InputTokens)
}

var tooManyInputTokensRegEx = regexp.MustCompile(`maximum context length is (\d+) tokens.*requested (\d+) tokens`) 


func togetherEmbed(toEmbed []string) ([][EMBED_DIM]float32, error) {
    if len(toEmbed) == 0 {
        return nil, fmt.Errorf("No strings given to embed. Length of toEmbed is 0.")
    }
	url := "https://api.together.xyz/v1/embeddings"

    // IF YOU CHANGE MODEL, BEWARE EMBED DIM
    payload := EmbeddingRequest{
        Model: "BAAI/bge-base-en-v1.5-vllm",
        Input: toEmbed,
    }
    body,_ := json.Marshal(payload)
    
    req,_ := http.NewRequest("POST", url, bytes.NewBuffer(body))
    req.Header.Set("Authorization", "Bearer " + TOGETHER_TOKEN)
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{}
    respRaw, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer respRaw.Body.Close()

    respBody, _ := io.ReadAll(respRaw.Body)

    if respRaw.StatusCode != http.StatusOK {
        tooManyMatch := tooManyInputTokensRegEx.FindStringSubmatch(string(respBody))
        if len(tooManyMatch) != 3 {
            err := fmt.Errorf("Status returned not ok in togetherEmbed: %d\nBody:\n%s", respRaw.StatusCode, respBody)
            return nil, err
        }
        maxTok, _ := strconv.Atoi(tooManyMatch[1])
        inpTok, _ := strconv.Atoi(tooManyMatch[2])

        err := &TooManyInputTokensError{
            MaxTokens: maxTok,
            InputTokens: inpTok,
        }

        return nil, err
	}

    type EmbeddingData struct {
        Embedding []float64 `json:"embedding"` // JSON numbers decoded as float64
        Index     int       `json:"index"`
        Object    string    `json:"object"`
    }

    type EmbeddingResponse struct {
        Data  []EmbeddingData `json:"data"`
        Model string          `json:"model"`
    }
    

    var resp EmbeddingResponse
    err = json.Unmarshal(respBody, &resp)
    if err != nil {
        return nil, err
    }


    embedding := make([][EMBED_DIM]float32, len(resp.Data))
    for i, e := range resp.Data {
        for j, val := range e.Embedding {
            embedding[i][j] = float32(val)
        }
    }

    return embedding,nil

}



func refreshSubjects(db *sql.DB) error {
    base := "http://export.arxiv.org/oai2"
    verb := "ListSets"

    // BEGIN FETCH SUBJECT LIST
    subjects := []ArxSubject{}
    for token := ""; ; {
        q := url.Values{"verb": {verb}}
        if token != "" {
            q.Set("resumptionToken", token)
        }
        u := base + "?" + q.Encode()

        resp, err := http.Get(u)
        if err != nil {
            return err
        }
        body, err := io.ReadAll(resp.Body)
        resp.Body.Close()
        if err != nil {
            return err
        }

        var doc OAI
        if err := xml.Unmarshal(body, &doc); err != nil {
            return err
        }
        if doc.Error != nil {
            return fmt.Errorf("oai error (%s): %s", doc.Error.Code, doc.Error.Text)
        }
        if doc.ListSets == nil {
            break
        }

        for _, s := range doc.ListSets.Sets {
            dotPath := strings.Split(s.Spec, ":")
            subj := ArxSubject{Name:s.Name}

            if len(dotPath) >= 2 && dotPath[0] == dotPath[1] {
                dotPath = dotPath[1:]
            }
            if len(dotPath) == 1 {
                subj.IsTop = true
            } else {
                subj.IsTop = false
                subj.Parent = strings.Join(dotPath[:len(dotPath)-1], ".")
            }
            subj.DotPath = strings.Join(dotPath, ".")
            
            subjects = append(subjects, subj)
        }

        token = doc.ListSets.ResumptionToken
        if token == "" {
            break
        }
    }    
    // END FETCH SUBJECT LIST

    // BEGIN ADD LIST TO DB
    tx, err := db.Begin()
    if err != nil {
        return err
    }

    stmt, err := tx.Prepare("INSERT OR IGNORE INTO categories (name, dot_path, parent, is_top) VALUES (?,?,?,?)")
    if err != nil {
        _ = tx.Rollback()
        return err
    }
    defer stmt.Close()

    for _,subj := range subjects {
        if _, err := stmt.Exec(subj.Name, subj.DotPath, subj.Parent, subj.IsTop); err != nil {
            _ = tx.Rollback()
            return err
        }
    }

    if err := tx.Commit(); err != nil {
        _ = tx.Rollback()
        return err
    }
    // END ADD LIST TO DB

    return nil
    
}


func pullPapers(db *sql.DB, dotPath string) error {
    resp, err := http.Get(ARXIV_RSS_URL + dotPath)
    if err != nil {
        return err
    }
    body, err := io.ReadAll(resp.Body)
    resp.Body.Close()
    if err != nil {
        return err
    }

    var rss RSS

    if err := xml.Unmarshal(body, &rss); err != nil {
        return err
    }

    tx, err := db.Begin()
    if err != nil {
        return err
    }
    
    // query to insert a new paper
    insertNewPaper, err := tx.Prepare(`
        INSERT OR IGNORE INTO papers (arxiv_id, guid, title, description, pub_date, creator)
        VALUES (?, ?, ?, ?, ?, ?)
        RETURNING id`)
    if err != nil {
        _ = tx.Rollback()
        return err
    }
    defer insertNewPaper.Close()
    
    
    // query to get the category id based on dotpath
    selectCategoryID, err := tx.Prepare(`SELECT id FROM categories WHERE dot_path = ?`)
    if err != nil {
        _ = tx.Rollback()
        return err
    }
    defer selectCategoryID.Close()
    
    // query to insert paper category
    insertPaperCategory, err := tx.Prepare(`
        INSERT OR IGNORE INTO paper_categories (paper_id, category_id)
        VALUES (?, ?)`)
    if err != nil {
        _ = tx.Rollback()
        return err
    }
    defer insertPaperCategory.Close()
    
    seenCategories := make(map[string]int)
    
    for _, item := range rss.Channel.Items {
        switch item.AnnounceType {
        case "new":
            arxivPrefixString := "arXiv:"
            arxivIdStart := strings.Index(item.Description, arxivPrefixString) + len(arxivPrefixString)
            space := strings.IndexRune(item.Description[arxivIdStart:], ' ') + arxivIdStart
            if space <= 6 {
                log.Println("bad description format for arXiv id:", item.Description)
                continue
            }
            arxivId := item.Description[arxivIdStart:space]

            abstractPrefixString := "Abstract: "
            abstractStart := strings.Index(item.Description, abstractPrefixString) + len(abstractPrefixString)

            t, err := time.Parse(RSS_TIME_LAYOUT_STR, item.PubDate)
            if err != nil {
                log.Println("ERROR PARSING TIME: ID: ", arxivId, " pubDate: ", item.PubDate)
                t = time.Now()
            }

            item.PubDate = t.UTC().Format(ISO_86O1_TIME_LAYOUT_STR)
    
            var paperId int
            if err := insertNewPaper.QueryRow(
                arxivId,
                item.GUID.Value,
                item.Title,
                item.Description[abstractStart:],
                item.PubDate,
                item.DCCreator,
            ).Scan(&paperId); err != nil {
                if errors.Is(err, sql.ErrNoRows) {
                    // INSERT was ignored, skip existing paper
                    continue
                } else {
                    _ = tx.Rollback()
                    return err
                }
            }
    
            for _, category := range item.Categories {
                categoryId, exists := seenCategories[category]
                if !exists {
                    if err := selectCategoryID.QueryRow(category).Scan(&categoryId); err != nil {
                        log.Println(category)
                        log.Println(err)
                        continue
                    }

                    seenCategories[category] = categoryId
                }
    
                if _, err := insertPaperCategory.Exec(paperId, categoryId); err != nil {
                    log.Println(err)
                    continue
                }
            }
        default:
            // ignore e.g. updates
        }
    }
    
    if err := tx.Commit(); err != nil {
        _ = tx.Rollback()
        return err
    }

    return nil
}



func writeEmbeds(db *sql.DB, paperIds []uint64, embeds [][EMBED_DIM]float32) error {
/*
    Write the embeddings for the given ids to the database
*/
    if len(paperIds) != len(embeds) {
        return fmt.Errorf("PaperIds and embeds given to writeEmbeds must be the same length (%d vs %d)", len(paperIds), len(embeds))
    }

    tx, err := db.Begin()
    if err != nil {
        return err
    }
    
    // query to insert a new paper
    insertNewEmbedding, err := tx.Prepare(`
        INSERT INTO papers_vec (paper_id, embedding) VALUES (?, ?)
    `)
    if err != nil {
        _ = tx.Rollback()
        return err
    }
    defer insertNewEmbedding.Close()

    for i, paperId := range paperIds {
        embedSerial, err := sqlite_vec.SerializeFloat32(embeds[i][:])
        if err != nil {
            _ = tx.Rollback()
            return err
        }
        if _, err = insertNewEmbedding.Exec(paperId, embedSerial); err != nil {
            _ = tx.Rollback()
            return err
        }
        
    }
    

    if err := tx.Commit(); err != nil {
        _ = tx.Rollback()
        return err
    }

    return nil
}


func generateMissingEmbeddings(db *sql.DB) {
    queryGetMissing := `
        SELECT id, title, description FROM papers 
        WHERE id NOT IN (
            SELECT paper_id FROM papers_vec
        )
    `

    row, err := db.Query(queryGetMissing)
    if err != nil {
        log.Fatal(err)
    }
    defer row.Close()

    
    paperIds := []uint64{} 
    toEmbed := []string{}
    for row.Next() {
        var paperId uint64
        var title string
        var description string
        
        err = row.Scan(&paperId, &title, &description)
        if err != nil {
            log.Println(err)
            continue
        }

        paperIds = append(paperIds, paperId)
        toEmbed = append(toEmbed, title + "\n" + description)
    }


    if len(paperIds) > CHUNK_SIZE * 2 {
        fmt.Printf("\nToo many embeddings (%d), generating in chunks (%d) (may be very slow): \n", len(paperIds), CHUNK_SIZE)

        tooBigEmbeds := []string{}
        tooBigIds := []uint64{}
        for i := 0; i < len(paperIds); i+=CHUNK_SIZE {
            chunkEnd := min(i + CHUNK_SIZE, len(paperIds))
            fmt.Printf("\r%3d%%", 100 * i / len(paperIds))
            chunkEmbeds, err := togetherEmbed(toEmbed[i: chunkEnd])
            chunkPaperIds := paperIds[i:chunkEnd]
            if err != nil {
                // Check if error NOT too many tokens
                if _,ok := err.(*TooManyInputTokensError); !ok {
                    log.Println(err.Error())
                    continue
                } 
                
                chunkEmbeds = [][EMBED_DIM]float32{}
                chunkPaperIds = []uint64{}
                // embed one-by-one to find the strings that were too big
                log.Println("Chunk has input(s) that is too big. Embedding one-by-one and isolating.")
                for j,emb := range(toEmbed[i:chunkEnd]) {
                    embVec, err := togetherEmbed([]string{emb})
                    if err != nil {
                        // check if too many tokens
                        if _,ok := err.(*TooManyInputTokensError); ok {
                            tooBigEmbeds = append(tooBigEmbeds, emb)
                            tooBigIds = append(tooBigIds, paperIds[i+j])
                        } else { 
                            log.Println(err.Error())
                        }
                        continue
                    }
                    chunkEmbeds = append(chunkEmbeds, embVec[0])
                    chunkPaperIds = append(chunkPaperIds, paperIds[i+j])
                }
            }

            // write embeds each chunk
            err = writeEmbeds(db, chunkPaperIds, chunkEmbeds)
            if err != nil {
                log.Println(err.Error())
                log.Println(i, chunkEnd)
                log.Println(len(toEmbed[i:chunkEnd]))
                log.Println("Continuing...")
                continue
            }
 
        }

        // Deal with too big embeds
        finishedEmbeds := [][EMBED_DIM]float32{}
        finishedIds := []uint64{}
        for len(tooBigEmbeds) > 0 {
            log.Printf("Dealing with too bigs (%d)\n", len(tooBigEmbeds))

            stillBigEmbeds := []string{}
            stillBigIds := []uint64{}
            for i, tooBig := range tooBigEmbeds {
                // truncate by 10%
                tooBig = tooBig[:len(tooBig) - len(tooBig) / 10]

                embVec,err := togetherEmbed([]string{tooBig})
                if err != nil {
                    // check if too many tokens
                    if _,ok := err.(*TooManyInputTokensError); ok {
                        stillBigEmbeds = append(stillBigEmbeds, tooBig)
                        stillBigIds = append(stillBigIds, tooBigIds[i])
                    } else { 
                        log.Println(err.Error())
                    }
                    continue
                }
                finishedEmbeds = append(finishedEmbeds, embVec[0])
                finishedIds = append(finishedIds, tooBigIds[i])
                
            }
            tooBigEmbeds = stillBigEmbeds
            tooBigIds = stillBigIds
        }


        // Write truncated embeds
        err = writeEmbeds(db, finishedIds, finishedEmbeds)
        if err != nil {
            log.Println(err.Error())
            log.Println("Continuing...")
        }
        
        fmt.Printf("\r%3d%%\n", 100)
    } else {
        fmt.Printf("Generating %d embeddings...\n", len(paperIds))
        goodPaperIds := paperIds
        embeds, err := togetherEmbed(toEmbed)
        tooBigEmbeds := []string{}
        tooBigIds := []uint64{}
        goodEmbeds := [][EMBED_DIM]float32{}
        goodPaperIds = []uint64{}
        if err != nil {
            // Check if error NOT too many tokens
            if _,ok := err.(*TooManyInputTokensError); !ok {
                log.Println(err.Error())
                // return without embedding or any such thing
                return
            } 

            for j,emb := range(toEmbed) {
                embVec, err := togetherEmbed([]string{emb})
                if err != nil {
                    // check if too many tokens
                    if _,ok := err.(*TooManyInputTokensError); ok {
                        tooBigEmbeds = append(tooBigEmbeds, emb)
                        tooBigIds = append(tooBigIds, paperIds[j])
                    } else { 
                        log.Println(err.Error())
                    }
                    continue
                }
                goodEmbeds = append(goodEmbeds, embVec[0])
                goodPaperIds = append(goodPaperIds, paperIds[j])
            }

            embeds = goodEmbeds
        }

        err = writeEmbeds(db, goodPaperIds, embeds)
        if err != nil {
            log.Println(toEmbed)
            log.Println(err.Error())
        }

        // Deal with too big embeds
        finishedEmbeds := [][EMBED_DIM]float32{}
        finishedIds := []uint64{}
        for len(tooBigEmbeds) > 0 {
            log.Printf("Dealing with too bigs (%d)\n", len(tooBigEmbeds))

            stillBigEmbeds := []string{}
            stillBigIds := []uint64{}
            for i, tooBig := range tooBigEmbeds {
                // truncate by 10%
                tooBig = tooBig[:len(tooBig) - len(tooBig) / 10]

                embVec,err := togetherEmbed([]string{tooBig})
                if err != nil {
                    // check if too many tokens
                    if _,ok := err.(*TooManyInputTokensError); ok {
                        stillBigEmbeds = append(stillBigEmbeds, tooBig)
                        stillBigIds = append(stillBigIds, tooBigIds[i])
                    } else { 
                        log.Println(err.Error())
                    }
                    continue
                }
                finishedEmbeds = append(finishedEmbeds, embVec[0])
                finishedIds = append(finishedIds, tooBigIds[i])
                
            }
            tooBigEmbeds = stillBigEmbeds
            tooBigIds = stillBigIds
        }


        // Write truncated embeds
        err = writeEmbeds(db, finishedIds, finishedEmbeds)
        if err != nil {
            log.Println(err.Error())
            log.Println("Continuing...")
        }
    }
    log.Println("Done.")


}


func keywordSearch(db *sql.DB, posWords string, negWords string, limit int) ([]PaperResult) {
    query := `
    WITH 
    pos AS 
        (SELECT rowid, bm25(papers_fts, 0.0, 1.0) AS score
            FROM papers_fts
            WHERE papers_fts MATCH :pos_words),
    neg AS 
        (SELECT pos.rowid, bm25(papers_fts, 0.0, 1.0) AS score
            FROM pos JOIN papers_fts ON pos.rowid = papers_fts.rowid
            WHERE :neg_words <> '' AND papers_fts MATCH :neg_words )
    SELECT p.arxiv_id, p.title, p.description, pos.score - :neg_weight * COALESCE(1.0 / (neg.score + 1e-9), 0.0) as final_score
        FROM pos 
        LEFT JOIN neg ON pos.rowid = neg.rowid 
        JOIN papers p ON pos.rowid = p.id
        ORDER BY final_score LIMIT :limit;
    `
    rows, err := db.Query(query, 
        sql.Named("pos_words", posWords),
        sql.Named("neg_words", negWords),
        sql.Named("neg_weight", 50),
        sql.Named("limit", limit),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer rows.Close()
    res := make([]PaperResult, limit)
    i := 0
    for rows.Next() {
        if err := rows.Scan(&res[i].ArxivId, &res[i].Title, &res[i].Desc, &res[i].Score); err != nil {
            log.Fatal(err)
        }
        i++
    }


    if err := rows.Err(); err != nil {
        log.Fatal(err)
    }
    
    return res
}


func semanticStringSearch(db *sql.DB, toEmbed string, limit int) ([]PaperResult) {
    embedVec, err := togetherEmbed([]string{toEmbed})
    if err != nil {
        log.Fatal(err)
    }

    queryVec,err := sqlite_vec.SerializeFloat32(embedVec[0][:])
    if err != nil {
        log.Fatal(err)
    }

    rows,err := db.Query(`
        SELECT arxiv_id, title, description, distance
        FROM papers JOIN (
            SELECT paper_id, vec_distance_cosine(embedding, ?) as distance FROM papers_vec
            ORDER BY distance
            LIMIT ?
        ) ON papers.id = paper_id
        ORDER BY distance
    `, queryVec, limit)
    if err != nil {
        log.Fatal(err)
    }
    defer rows.Close()


    res := make([]PaperResult, limit)    
    i := 0
    for rows.Next() {
        if err := rows.Scan(&res[i].ArxivId, &res[i].Title, &res[i].Desc, &res[i].Score); err != nil {
            log.Fatal(err)
        }
        i++
    } 

    return res
}



func pullAllNewPapers(db *sql.DB) error {
    rows, err := db.Query("SELECT name, dot_path FROM categories WHERE is_top=true", nil)
    if err != nil {
        return err
    }
    defer rows.Close()

    subjNames := []string{}
    subjDotPaths := []string{}
    for rows.Next() {
        var name string
        var dotPath string
        if err := rows.Scan(&name, &dotPath); err != nil {
            return err
        }

        subjNames = append(subjNames, name)
        subjDotPaths = append(subjDotPaths, dotPath)
    }

    for _,subjPath := range subjDotPaths {
        err := pullPapers(db, subjPath)
        if err != nil {
            return err
        }
    }
    return nil
}


///--- BEGIN auth ---///
type JWTClaims struct {
    UserId uint64 `json:"userid"`
    jwt.RegisteredClaims
}


func createAndSetToken(w http.ResponseWriter, userId uint64) error {
    expirationTime := time.Now().Add(8 * time.Hour)
    claims := &JWTClaims{
        UserId: userId,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(expirationTime),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            NotBefore: jwt.NewNumericDate(time.Now()),
        },
    }

    token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
    tokenStr, err := token.SignedString(JWT_KEY)
    if err != nil {
        return fmt.Errorf("Error signing token string: %v", err)
    }

    http.SetCookie(w, &http.Cookie{
        Name:     "token",
        Value:    tokenStr,
        Expires:  expirationTime,
        HttpOnly: true,
        Path: "/",
        // Secure: true TODO when https set up
    })

    return nil
}   


func redirect(w http.ResponseWriter, r *http.Request, redirectURL string) {
    if r.Header.Get("HX-Request") == "true" {
        w.Header().Set("HX-Redirect", redirectURL)
        w.WriteHeader(http.StatusOK)
        return
    }
    http.Redirect(w, r, redirectURL, http.StatusSeeOther)
}


func authenticate(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        cookie, err := r.Cookie("token")
        if err != nil {
            redirect(w, r, "/login")
            return
        }

        ///--- verify jwt ---///
        claims := &JWTClaims{}

        token, err := jwt.ParseWithClaims(cookie.Value, claims, func(token *jwt.Token) (interface{}, error) {
            if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
                return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
            }
            return JWT_KEY, nil
        })

        if err != nil {
            log.Println(err.Error())
            redirect(w, r, "/login")
            return
        }
        
        if !token.Valid {
            log.Println("invalid token")
            redirect(w, r, "/login")
            return
        }
        
        ctx := context.WithValue(r.Context(), USER_ID_CONTEXT_KEY, uint64(claims.UserId))
        next.ServeHTTP(w,r.WithContext(ctx))
    })
}



type BaseTemplateData struct {
    BasePath string
    Title string
}

func getBaseData(r *http.Request, title string) BaseTemplateData {
    // Pass if proxied, ensures /static points to right place behind proxy
    basePath := r.Header.Get("SCRIPT_NAME")

    if basePath == "/" {
        basePath = ""
    }

    data := BaseTemplateData{
        BasePath: basePath,
        Title: title,
    }

    return data
}


func main() {
    log.SetFlags(log.Lshortfile)

    sqlite_vec.Auto()
    db, err := sql.Open("sqlite3", "./app.db")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()


    err = godotenv.Load()
    if err != nil {
        log.Fatal(err)
    }
    JWT_KEY = []byte(os.Getenv("JWT_SECRET_KEY"))
    TOGETHER_TOKEN = os.Getenv("TOGETHER_TOKEN")
    

    shouldRefreshSubjectsPtr := flag.Bool("s", false, "Pull the latest subjects (e.g. Physics) from arxiv. Should rarely, if ever, be needed after the first run.")
    shouldPullPapersPtr := flag.Bool("pull", false, "Pull todays papers from arxiv. Should be run daily.")
    shouldEmbedPtr := flag.Bool("embed", false, "Generate embeddings for all the papers in the db that don't already have them. This can be quite slow. Papers without embeddings will not be returned by vector queries.")
    shouldKeywordSearchPtr := flag.Bool("k", false, "Search on keywords using bm25. Can have two string inputs (1) POSITIVE and (2) NEGATIVE keywords. Use word1 OR word2 for union of terms.")
    shouldVectorSearchPtr := flag.Bool("v", false, "Search on vector embeddings. Use a string similar to the kind of paper you want to see. E.g. 'New attention mechanism in transformers.'")
    numResultsPtr := flag.Int("n", 5, "Number of results to return for the query.")
    portPtr := flag.String("p", "8080", "Port to run the server on.")
    noServerPtr := flag.Bool("ns", false, "Set to not run the server. (i.e. only run the commands given by other flags)")

    flag.Parse()

    numResults := *numResultsPtr


    if *shouldRefreshSubjectsPtr {
        refreshSubjects(db)
    }

    // Pull todays papers from top level subjects
    if *shouldPullPapersPtr {
        err := pullAllNewPapers(db)
        if err != nil {
            log.Fatal(err)
        }
    }

    if *shouldEmbedPtr {
        generateMissingEmbeddings(db)
    }


    searchStr := flag.Arg(0)
    var negStr string
    if flag.NArg() != 2 {
        negStr = ""
    } else {
        negStr = flag.Arg(1)
    }
    
    if *shouldKeywordSearchPtr {
        res := keywordSearch(db, searchStr, negStr, numResults)
        fmt.Printf("Keyword search results:\n")
        for _,v := range res {
            fmt.Printf("%s (%s): %.3f\n", v.Title, v.ArxivId, v.Score)
        }
        fmt.Printf("\n")
    }

    if *shouldVectorSearchPtr {
        fmt.Printf("Vector search results:\n")
        res := semanticStringSearch(db, searchStr, numResults)
        for _,v := range res {
            fmt.Printf("%s (%s): %.3f\n", v.Title, v.ArxivId, v.Score)
        }
        fmt.Printf("\n")
    }


    if *noServerPtr {
        return
    }

    port := *portPtr


    fs := http.FileServer(http.Dir("./static"))
    http.Handle("/static/", http.StripPrefix("/static/", fs))


    tmpl := template.Must(template.ParseFiles("templates/index.html"))
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        data := getBaseData(r, "Arxiv Feed")
        if err := tmpl.ExecuteTemplate(w, "index", data); err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
    })


    http.HandleFunc("/signup", func(w http.ResponseWriter, r *http.Request) {
        data := getBaseData(r, "Signup")
        if err := tmpl.ExecuteTemplate(w, "signup", data); err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
    })


    http.HandleFunc("/login", func(w http.ResponseWriter, r *http.Request) {
        data := getBaseData(r, "login")
        if err := tmpl.ExecuteTemplate(w, "login", data); err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
    })


    http.HandleFunc("/api/signup", func(w http.ResponseWriter, r *http.Request) {
        email := r.FormValue("email")
        password := r.FormValue("password")
        passConf := r.FormValue("passConf")

        if password != passConf {
            http.Error(w, "Passwords do not match", http.StatusUnprocessableEntity)
        }

        phash, err := bcrypt.GenerateFromPassword([]byte(password), BCRYPT_STRENGTH)

        if err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return 
        }

        query := `
            INSERT INTO users (email, phash)
            VALUES (?, ?)
        `

        var userId uint64
        res, err := db.ExecContext(r.Context(), query, email, phash)
        if err != nil {
          var sqlErr sqlite3.Error
          if errors.As(err, &sqlErr) && (sqlErr.Code == sqlite3.ErrConstraint || sqlErr.ExtendedCode == sqlite3.ErrConstraintUnique) {
            log.Println("Email already exists")
            redirect(w, r, "/login")
            return
          }
          log.Println(err.Error())
          http.Error(w, err.Error(), http.StatusInternalServerError)
          return
        }

        lastId, err := res.LastInsertId()
        if err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        userId = uint64(lastId)


        err = createAndSetToken(w, userId)
        if err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
        
        redirect(w,r,"/")
    })


    http.HandleFunc("/api/login", func(w http.ResponseWriter, r *http.Request) {
        email := r.FormValue("email")
        password := r.FormValue("password")

        var userId uint64
        var phash string

        query := "SELECT id, phash FROM users WHERE email = $1"

        err := db.QueryRowContext(r.Context(), query, email).Scan(&userId, &phash)
        if err != nil {
            if err == sql.ErrNoRows {
                redirect(w, r, "/signup")
            } else {
                log.Println(err.Error())
                http.Error(w, err.Error(), http.StatusInternalServerError)
            }
            return
        }


        err = bcrypt.CompareHashAndPassword([]byte(phash), []byte(password))
        if err != nil {
            if errors.Is(err, bcrypt.ErrMismatchedHashAndPassword) {
                log.Println("Passwords don't match")
                http.Error(w, err.Error(), http.StatusUnauthorized)
                return
            }
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }

        createAndSetToken(w, userId)
        redirect(w, r, "/")
    })

    http.HandleFunc("/api/keyword-search", func (w http.ResponseWriter, r *http.Request) {
        if err := r.ParseForm(); err != nil {
            http.Error(w, "bad form", http.StatusBadRequest)
		    return
	    }
        query := r.FormValue("searchQuery")
        results := keywordSearch(db, query, "", 5)

        if err := tmpl.ExecuteTemplate(w, "paperResults", results); err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
    })


    http.Handle("/api/semantic-search", authenticate(http.HandlerFunc(func (w http.ResponseWriter, r *http.Request) {
        if err := r.ParseForm(); err != nil {
            http.Error(w, "bad form", http.StatusBadRequest)
		    return
	    }
        query := r.FormValue("searchQuery")
        results := semanticStringSearch(db, query, 5)

        if err := tmpl.ExecuteTemplate(w, "paperResults", results); err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
        }
    })))


    http.HandleFunc("/api/refresh-subjects", func (w http.ResponseWriter, r *http.Request) {
        err := refreshSubjects(db)               
        if err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        fmt.Fprintf(w, "Subjects Refreshed.")
    })

    http.HandleFunc("/api/pull-papers", func (w http.ResponseWriter, r *http.Request) {
        err := pullAllNewPapers(db)               
        if err != nil {
            log.Println(err.Error())
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }
        fmt.Fprintf(w, "New papers pulled, only keyword search is available until the embeddings are generated.")
    })




    log.Println("Listening on port " + port)
    if err := http.ListenAndServe(":" + port, nil); err != nil {
        log.Fatal(err)
    }

}
