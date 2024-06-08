# Holding - want to replace the reid function with one where it only compares interested_ids for reid

def reid(self, img, interested_ids):
        """Reid the tracked object."""
        outputs = self.extract_features(img)
        faiss.normalize_L2(outputs)
        print('interested_ids : ', interested_ids)
        vectors = np.array([self.faiss.reconstruct(self.id_to_image[id]) for id in interested_ids])
        print('what is this id image thing???? s', self.id_to_image[id[0]][0])
        vectors = np.array([self.faiss.reconstruct(id) for id in interested_ids])
        new_index = faiss.IndexFlatL2(vectors.shape[1])
        new_index.add(vectors)
        D, I = new_index.search(outputs, 5)
        weights = D.max() - D
        weights = weights / weights.sum()

        # Multiply each ID by its corresponding weight to get the weighted IDs
        weighted_ids = I * weights

        # Sum the weighted IDs to get the final ID
        final_id = weighted_ids.sum()

        # Round the final ID to the nearest integer
        final_id = round(final_id)
        print(f"Final ID: {final_id}")


        # _, I = self.faiss.search(outputs, 5)
        print(f"Indices: {I}")
        print('self.lost_stracks', self.lost_stracks)
        # combined_stracks = self.tracked_stracks + self.lost_stracks
        # for stracks in combined_stracks:
        #     print('stracks.idx', stracks.track_id)
        #     first_five = self.id_to_image[stracks.track_id][:5]
            
            # # Display the queried image
            # plt.figure(figsize=(10, 2))
            # plt.subplot(1, 6, 1)
            # plt.imshow(img)  # Make sure 'img' is defined and is the queried image
            # plt.title("Queried Image")

            # for i, closest_img in enumerate(first_five, 2):  # Start enumeration from 2
            #     i = 2 if i == 0 else i
            #     print('what is i?', i)
            #     plt.subplot(1, 6, i)
            #     plt.imshow(closest_img)
            #     plt.title(f"Idx {stracks.track_id}")
            # plt.show()    

        # Assuming you have a dictionary `id_to_image` mapping IDs to images
        closest_images = [self.id_to_image[id][0] for id in I[0]]

        # Display the queried image
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 6, 1)
        plt.imshow(img)
        plt.title("Queried Image")

        # Display the top 5 closest images
        for i, closest_img in enumerate(closest_images,2 ):
            plt.subplot(1, 6, i)
            plt.imshow(closest_img)
            plt.title(f"Closest {i-1}")

        plt.show()
        return I
