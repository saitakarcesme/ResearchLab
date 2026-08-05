# Activity redesign plan

## Neden önceki değişiklikler aynı göründü?

Mevcut `LiveLogs` bileşeni ilk günden beri aynı temel bilgi mimarisini koruyor: üstte tek bir “şu an” satırı, altında zaman sıralı küçük log satırları. Sonraki düzenlemeler tekrarları grupladı, açıklamaları değiştirdi ve bazı satırları gizledi; ancak DOM yapısı, görsel hiyerarşi ve kullanıcının tarama biçimi değişmedi. Bu yüzden sonuç yeni bir Activity deneyimi değil, aynı log listesinin farklı bir teması oldu.

## Yeni yaklaşım: deney günlüğü değil, karar akışı

Varsayılan görünüm kronolojik log olmayacak. Activity dört katmandan oluşacak:

1. **Aşama şeridi** — `Planla → Değiştir → Ölç → Karar ver → Yayınla`. Tamamlanan, aktif ve sıradaki aşama tek bakışta seçilecek.
2. **Şu an yapılan iş** — Tek bir doğal cümle, deney numarası ve geçen süre. Alt açıklama veya küçük teknik metin yığını olmayacak.
3. **Karar kartları** — Yalnızca sonucu değiştiren kabul, anlamlı ret, hata ve kullanıcı kontrolü kart olarak eklenecek. Arka arkaya aynı sonucu veren deneyler tek kartta “12 varyasyon denendi; en yakını …” biçiminde birleşecek.
4. **Teknik kayıt çekmecesi** — Ham olay akışı varsayılan ekranda görünmeyecek. Gerektiğinde açılan ayrı bir panelde aranabilir ve kopyalanabilir olacak.

## Kart türleri

- **İyileştirme:** Eski değer → yeni değer, yüzde farkı ve bunu sağlayan değişikliğin insan diliyle tek cümlesi.
- **Öğrenilen sınır:** Birden çok reddedilen deneyi tek sonuca bağlayan kısa çıkarım.
- **Engel:** Hatanın kullanıcı etkisi ve sistemin sıradaki davranışı; stack trace değil.
- **Kontrol:** Başlatıldı, duraklatıldı, devam etti ve durduruldu olayları küçük zaman işaretleri.

## Veri dönüşümü

Backend ham olayları saklamaya devam edecek. Yeni bir `activity-summary` katmanı olayları kararlı kimliklerle şu gruplara çevirecek: `phase`, `current_action`, `milestone`, `lesson`, `incident`, `control`. Aynı deney veya aynı hata ailesi yeniden geldiğinde yeni satır eklemek yerine mevcut özet güncellenecek.

## Etkileşim

- Progress grafiğindeki kabul noktasına tıklamak ilgili iyileştirme kartını vurgulayacak.
- Karar kartı grafikteki deney aralığını vurgulayacak.
- “Teknik kayıt” çekmecesi Activity yüksekliğini büyütmeyecek; kendi içinde scroll olacak.
- TV modunda yalnızca aşama, güncel iş ve son önemli karar gösterilecek.

## Kabul ölçütleri

- İlk bakışta üç saniye içinde “şu an ne yapıyor?” ve “son önemli sonuç neydi?” anlaşılmalı.
- Varsayılan görünümde en fazla beş karar kartı bulunmalı.
- Tekrarlanan 100 başarısız deney, 100 satır üretmemeli.
- 10 piksel açıklama metni veya yan yana sıkıştırılmış teknik metadata olmamalı.
- Ham logların hiçbiri kaybolmamalı; yalnızca ayrı teknik çekmeceye taşınmalı.
- Değişiklik, mevcut `LiveLogs` satır yapısını yeniden stillendirmekle yapılmamalı; yeni bir veri modeli ve yeni component ağacı kullanılmalı.
